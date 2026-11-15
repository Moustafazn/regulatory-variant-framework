"""Download and look up local AlphaGenome AVI scores for Ho, Bourges and Siraj.

Run the lookup after placing the archive in data/alphagenome_avi/:
    venv/bin/python scripts/download_alphagenome_avi.py
Download/resume the archive with the same script:
    venv/bin/python scripts/download_alphagenome_avi.py --download-archive
Select datasets with --datasets ho bourges or --datasets siraj.

The script is keyless. The downloaded ZIP is read directly: only its small Tabix index is extracted.
The 88 GB score file stays inside the ZIP; no second copy is created. This works
with the official ZIP's stored (not ZIP-deflated) members. An already unpacked
.tsv.gz also works. Archive, hg38 reference, index, coordinate map, progress ledger,
and extracted tables live in data/alphagenome_avi/. Dependencies: pandas and
biopython; pysam is needed only for an already unpacked .tsv.gz. Outputs are AVI
impact scores, not cell-type-specific track features. Indels are unsupported.

Sources checked:
https://github.com/google-deepmind/alphagenome_research
  (model inference code, not an AVI archive reader)
https://github.com/Ensembl/VEP_plugins/blob/release/116/AVI.pm
  (CHROM, POS, REF, ALT, raw_score, PHRED; positions are 1-based)
https://pysam.readthedocs.io/en/latest/api.html#tabix-files
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import shutil
import ssl
import struct
import sys
import tempfile
import time
import zipfile
import zlib
from urllib.request import Request, urlopen

import truststore

import pandas as pd
from Bio import Align, SeqIO, bgzf

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'output/experiments/regulatory-variant-framework'
AVI_DIR = ROOT / 'data/alphagenome_avi'
AVI_URL = 'https://deepmind.google.com/science/alphagenome/_/download/atlas/avi_scores_snvs_tabix.zip'
COMPLEMENT = str.maketrans('ACGT', 'TGCA')
CHAIN_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz'
SEARCH_RADIUS = 50000
MAX_FLANK_MISMATCHES = 3
BOURGES_MAP_VERSION = 'bourges-hg38-oligos-v1'
SIRAJ_BUILD_SOURCE = (
    'https://github.com/julirsch/finemapped_mpra/blob/'
    '1e38d1e3d4414dfa3a62b22110eba0136eda197d/code/preprocess/preprocess_enformer.R#L74-L89'
)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verified_tls_context():
    """Use the OS trust store while retaining normal TLS verification."""
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def download_archive(path):
    """Download or resume the official archive without extracting its 88 GB member."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and zipfile.is_zipfile(path):
        print(f'AVI archive already exists: {path}', flush=True)
        return
    partial = path.with_name(path.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(AVI_URL, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urlopen(request, timeout=120, context=verified_tls_context()) as response:
        status = getattr(response, 'status', None)
        content_range = response.headers.get('Content-Range', '')
        if offset and status == 206 and content_range.startswith(f'bytes {offset}-'):
            mode = 'ab'
            print(f'Resuming AVI archive at {offset:,} bytes.', flush=True)
        else:
            mode = 'wb'
            if offset:
                print('Server did not resume the partial archive; restarting it.', flush=True)
            offset = 0
        length = response.headers.get('Content-Length')
        total = offset + int(length) if length and length.isdigit() else None
        written = offset
        last_report = time.monotonic()
        with partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                written += len(chunk)
                if time.monotonic() - last_report >= 15:
                    size = f'{written / 1e9:.1f} GB'
                    if total:
                        size += f' / {total / 1e9:.1f} GB ({written / total:.1%})'
                    print(f'AVI archive download: {size}', flush=True)
                    last_report = time.monotonic()
            handle.flush()
    if not zipfile.is_zipfile(partial):
        raise ValueError(f'Download is incomplete or not a ZIP: {partial}. Rerun --download-archive to resume.')
    partial.replace(path)
    print(f'Downloaded AVI archive: {path} ({path.stat().st_size / 1e9:.2f} GB)', flush=True)


def chromosome(value):
    name = str(value).removeprefix('chr')
    return 'chr' + ('M' if name == 'MT' else name)


class LocalProgress:
    """Durable unique-variant ledger; separate from all existing track caches."""

    def __init__(self, path, atlas, index):
        # Tie reuse to this local Atlas release and index. Do not hash 88 GB on
        # every restart. A replaced file/index starts a separate lookup namespace.
        identity = [atlas.stat().st_size, atlas.stat().st_mtime_ns, sha256(index)]
        if atlas.suffix == '.zip':
            with zipfile.ZipFile(atlas) as archive:
                members = [item for item in archive.infolist()
                           if item.filename.endswith(('.tsv.gz', '.tsv.bgz'))]
                if len(members) != 1:
                    raise ValueError(f'Expected one AVI score member; found {len(members)}.')
                member = members[0]
                identity.extend((member.filename, member.CRC, member.file_size))
        self.source = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.connection = sqlite3.connect(path)
        self.connection.execute('''CREATE TABLE IF NOT EXISTS scores (
            source TEXT, chrom TEXT, position INTEGER, ref TEXT, alt TEXT,
            raw REAL, phred REAL, status TEXT,
            PRIMARY KEY (source, chrom, position, ref, alt))''')
        self.values = {tuple(row[:4]): tuple(row[4:]) for row in self.connection.execute(
            'SELECT chrom, position, ref, alt, raw, phred, status FROM scores WHERE source=?', (self.source,))}
        self.pending = []
        print(
            'AVI records already processed for this exact archive and index: '
            f'{len(self.values):,}',
            flush=True,
        )

    def save(self, key, value):
        self.values[key] = value
        self.pending.append((self.source, *key, *value))
        if len(self.pending) >= 1000:
            self.flush()

    def flush(self):
        if self.pending:
            self.connection.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)', self.pending)
            self.connection.commit()
            self.pending.clear()

    def close(self):
        try:
            self.flush()
        finally:
            self.connection.close()


def resolve_atlas(path, index_dir=None):
    """Keep the large ZIP member in place; extract only the small .tbi index."""
    path = Path(path).expanduser().resolve()
    if path.suffix in {'.crdownload', '.part', '.download'}:
        raise ValueError('The download is incomplete. Wait for the final ZIP filename.')
    if not path.exists():
        raise FileNotFoundError(f'Atlas not found: {path}. Wait for the download or set --atlas.')
    if path.suffix == '.zip':
        if not zipfile.is_zipfile(path):
            raise ValueError('ZIP is incomplete or invalid; wait for the download to finish.')
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if n.endswith(('.tsv.gz', '.tsv.bgz'))]
            if len(candidates) != 1:
                raise ValueError(f'Expected one AVI TSV in ZIP; found {len(candidates)}.')
            member = candidates[0]
            info = archive.getinfo(member)
            if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
                raise ValueError('Direct reading requires an unencrypted, ZIP_STORED score member, as in the official ZIP.')
            if member + '.tbi' not in archive.namelist():
                raise ValueError('Direct ZIP reading requires the matching .tbi index.')
            small_index = archive.getinfo(member + '.tbi')
            destination = Path(index_dir) if index_dir else AVI_DIR
            destination.mkdir(parents=True, exist_ok=True)
            index = destination / f'{info.CRC:08x}-{small_index.CRC:08x}.tbi'
            if not index.is_file() or zlib.crc32(index.read_bytes()) != small_index.CRC:
                # Reading this 3.2 MB member also checks its ZIP checksum.
                index.write_bytes(archive.read(small_index))
            print(f'Reading ZIP directly; only {small_index.file_size / 1e6:.1f} MB of index extracted.', flush=True)
            return path, index
    elif path.is_dir():
        candidates = sorted(path.glob('*.tsv.gz')) + sorted(path.glob('*.tsv.bgz'))
        if len(candidates) != 1:
            raise ValueError('Pass the exact AVI .tsv.gz filename with --atlas.')
        path = candidates[0]
    index = next((Path(str(path) + suffix) for suffix in ('.tbi', '.csi')
                  if Path(str(path) + suffix).is_file()), None)
    if index is None:
        raise FileNotFoundError(f'Missing {path.name}.tbi/.csi; extract the supplied index too.')
    return path, index


class StoredMember:
    """Seekable, bounded view of a ZIP_STORED member, without copying its bytes."""

    def __init__(self, archive_path, member):
        self.handle = archive_path.open('rb')
        try:
            self.handle.seek(member.header_offset)
            header = self.handle.read(30)
            signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack('<4s5H3I2H', header)
            if signature != b'PK\x03\x04' or method != 0 or flags & 1:
                raise ValueError('Invalid or compressed local ZIP member header.')
            self.start = member.header_offset + 30 + name_size + extra_size
            self.size = member.file_size
            if member.compress_size != self.size or self.start + self.size > archive_path.stat().st_size:
                raise ValueError('Truncated or inconsistent ZIP member.')
            self.handle.seek(self.start)
        except BaseException:
            self.handle.close()
            raise

    def tell(self):
        return self.handle.tell() - self.start

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek origin')
        position = offset + (0 if whence == 0 else self.tell() if whence == 1 else self.size)
        if not 0 <= position <= self.size:
            raise ValueError('Seek outside the stored score file')
        self.handle.seek(self.start + position)
        return position

    def read(self, size=-1):
        remaining = self.size - self.tell()
        return self.handle.read(remaining if size < 0 else min(size, remaining))

    def close(self):
        self.handle.close()


class ZipTabix:
    """AVI SNV Tabix lookup inside the official ZIP, using bounded BGZF seeks.

    Index layout/binning: https://samtools.github.io/hts-specs/tabix.pdf
    BGZF decoding and block CRC checks are handled by Biopython. Cache the last
    16 kb genomic window in RAM so nearby variants do not rescan the same block.
    No chromosome or score file is expanded to disk.
    """

    def __init__(self, path, index):
        data = gzip.decompress(index.read_bytes())
        offset = 0

        def unpack(fmt):
            nonlocal offset
            values = struct.unpack_from(fmt, data, offset)
            offset += struct.calcsize(fmt)
            return values

        magic, n_ref, fmt, seq, begin, end, meta, skip, name_size = unpack('<4s8i')
        if magic != b'TBI\x01' or (fmt, seq, begin, end, meta, skip) not in (
                (0, 1, 2, 2, 35, 0), (0, 1, 2, 0, 35, 0), (2, 1, 2, 0, 35, 0)):
            raise ValueError('Unsupported Tabix layout; expected the AVI SNV table.')
        self.contigs = tuple(data[offset:offset + name_size].decode().rstrip('\0').split('\0'))
        offset += name_size
        if len(self.contigs) != n_ref:
            raise ValueError('Invalid Tabix chromosome list')
        self.regions = {}
        for chrom in self.contigs:
            bins = {}
            for _ in range(unpack('<i')[0]):
                bin_id, n_chunks = unpack('<Ii')
                bins[bin_id] = [unpack('<QQ') for _ in range(n_chunks)]
            n_linear = unpack('<i')[0]
            self.regions[chrom] = (bins, unpack(f'<{n_linear}Q'))
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if m.filename.endswith(('.tsv.gz', '.tsv.bgz'))]
            if len(members) != 1 or members[0].compress_type != zipfile.ZIP_STORED:
                raise ValueError('Expected one stored AVI score member')
            view = StoredMember(path, members[0])
        try:
            self.reader = bgzf.BgzfReader(fileobj=view, mode='rt', max_cache=32)
            self.header = []
            while True:
                line = self.reader.readline()
                if not line.startswith('#'):
                    break
                self.header.append(line.rstrip('\r\n'))
        except BaseException:
            view.close()
            raise
        self.cached_window = None
        self.positions, self.lines = [], []

    def _read_window(self, chrom, number):
        start, stop = number << 14, (number + 1) << 14
        bins, linear = self.regions[chrom]
        bin_ids = [0]
        for shift, base in ((26, 1), (23, 9), (20, 73), (17, 585), (14, 4681)):
            bin_ids.extend(range(base + (start >> shift), base + ((stop - 1) >> shift) + 1))
        minimum = linear[number] if number < len(linear) else 0
        chunks = sorted((max(a, minimum), b) for bin_id in bin_ids
                        for a, b in bins.get(bin_id, ()) if b > minimum)
        merged = []
        for a, b in chunks:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        positions, lines = [], []
        for a, b in merged:
            self.reader.seek(a)
            while self.reader.tell() < b:
                line = self.reader.readline()
                if not line:
                    raise ValueError('Unexpected end of score member while reading an indexed chunk')
                if line.startswith('#'):
                    continue
                actual_chrom, pos, _ = line.split('\t', 2)
                if actual_chrom != chrom:
                    continue
                position = int(pos) - 1
                if position >= stop:
                    return positions, lines
                if position >= start:
                    positions.append(position)
                    lines.append(line.rstrip('\r\n'))
        return positions, lines

    def fetch(self, chrom, start, end):
        if start < 0 or end <= start:
            raise ValueError('Invalid query interval')
        for number in range(start >> 14, ((end - 1) >> 14) + 1):
            key = (chrom, number)
            if key != self.cached_window:
                positions, lines = self._read_window(chrom, number)
                self.positions, self.lines, self.cached_window = positions, lines, key
            lo, hi = bisect_left(self.positions, start), bisect_left(self.positions, end)
            yield from self.lines[lo:hi]

    def close(self):
        self.reader.close()


def lift_snvs(frame, chain):
    """Map requested 1-based GRCh37 SNVs only; retain ambiguous mappings."""
    positions = defaultdict(set)
    for row in frame.itertuples(index=False):
        positions[chromosome(row.chrom)].add(int(row.position) - 1)
    positions = {c: sorted(p) for c, p in positions.items()}
    hits = defaultdict(set)
    with gzip.open(chain, 'rt') as handle:
        for alignment in Align.parse(handle, 'chain'):
            chrom = alignment.target.id
            if chrom not in positions:
                continue
            source, target = alignment.coordinates
            for a, b, c, d in zip(source[:-1], source[1:], target[:-1], target[1:]):
                if b <= a or abs(d - c) != b - a:
                    continue  # A gap cannot map a single base.
                for position in positions[chrom][bisect_left(positions[chrom], a):
                                                 bisect_left(positions[chrom], b)]:
                    mapped = c + position - a if d > c else c - (position - a) - 1
                    hits[(chrom, position + 1)].add(
                        (alignment.query.id, int(mapped) + 1, '+' if d > c else '-'))
    return hits


def ensure_chain(path):
    """Fetch the small UCSC coordinate chain once, next to the AVI resources."""
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading UCSC hg19-to-hg38 coordinate map...', flush=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(
                CHAIN_URL,
                timeout=60,
                context=verified_tls_context(),
            ) as response:
                shutil.copyfileobj(response, handle)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        with gzip.open(temporary, 'rb') as handle:
            if not handle.read(6).startswith(b'chain '):
                raise ValueError('Invalid UCSC chain download')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _paired_oligos(raw, row):
    windows = []
    for _, group in raw[raw.rsID.eq(row.rsid)].groupby('elementTag'):
        alleles = {allele: group.loc[group.allele.eq(allele), 'genomicSequence'].unique()
                   for allele in ('ref', 'alt')}
        if any(len(values) != 1 for values in alleles.values()):
            raise ValueError('Missing or conflicting source allele sequences')
        reference, alternate = (str(alleles[allele][0]).upper() for allele in ('ref', 'alt'))
        differences = [i for i, (a, b) in enumerate(zip(reference, alternate)) if a != b]
        if len(reference) != len(alternate) or len(differences) != 1:
            raise ValueError('Source window is not a single-nucleotide allele pair')
        offset = differences[0]
        if (reference[offset], alternate[offset]) != (row.ref, row.alt):
            raise ValueError('Prepared alleles disagree with the source sequences')
        if set(reference + alternate) - set('ACGT'):
            raise ValueError('Source sequence contains ambiguous bases')
        windows.append((reference, alternate, offset))
    if not windows:
        raise ValueError('No source oligos for this rsID')
    return windows


def _oriented(window, strand):
    reference, alternate, offset = window
    if strand == '-':
        return (reference.translate(COMPLEMENT)[::-1],
                alternate.translate(COMPLEMENT)[::-1], len(reference) - 1 - offset)
    return window


def _match_oligos(sequence, windows):
    """Find one best match for paired assay windows within the lifted locus."""
    central = max(windows, key=lambda w: min(w[2], len(w[0]) - w[2] - 1))
    candidates = []
    for strand in ('+', '-'):
        reference, _, offset = _oriented(central, strand)
        if min(offset, len(reference) - offset - 1) < 30:
            raise ValueError('Source windows lack 30-base flanks on both sides')
        positions = set()
        for seed_offset in (offset - 30, offset + 1):
            seed = reference[seed_offset:seed_offset + 30]
            start = sequence.find(seed)
            while start != -1:
                positions.add(start - seed_offset + offset)
                start = sequence.find(seed, start + 1)
        for position in sorted(positions):
            mismatches = []
            for window in windows:
                ref, alt, index = _oriented(window, strand)
                start = position - index
                end = start + len(ref)
                if start < 0 or end > len(sequence):
                    break
                actual = sequence[start:end]
                count = sum(a != b for i, (a, b) in enumerate(zip(ref, actual)) if i != index)
                if count > MAX_FLANK_MISMATCHES or actual[index] not in (ref[index], alt[index]):
                    break
                mismatches.append(count)
            else:
                ref, alt, index = _oriented(central, strand)
                assay_ref, assay_alt = ref[index], alt[index]
                genomic_ref = sequence[position]
                swapped = genomic_ref == assay_alt
                candidates.append({
                    'position': position + 1, 'strand': strand,
                    'ref': genomic_ref, 'alt': assay_ref if swapped else assay_alt,
                    'assay_ref_forward': assay_ref, 'assay_alt_forward': assay_alt,
                    'score_multiplier': -1 if swapped else 1,
                    'flank_mismatches_by_window': mismatches,
                })
    if candidates:
        best = min(sum(item['flank_mismatches_by_window']) for item in candidates)
        candidates = [item for item in candidates
                      if sum(item['flank_mismatches_by_window']) == best]
    if len(candidates) != 1:
        raise ValueError(f'Expected one best sequence-verified GRCh38 locus; found {len(candidates)}')
    return candidates[0]


def prepare_bourges_map(args, reference):
    """Build or reuse a sequence-verified Bourges hg19-to-hg38 map."""
    source = args.prepared_dir / 'bourges.csv.gz'
    if not source.is_file():
        raise FileNotFoundError(f'Prepared Bourges data not found: {source}')
    if not args.bourges_oligos.is_file():
        raise FileNotFoundError(f'Bourges source oligos not found: {args.bourges_oligos}')
    ensure_chain(args.chain)
    frame = pd.read_csv(source, dtype={'chrom': str})
    required = {'example_id', 'rsid', 'chrom', 'position', 'ref', 'alt', 'genome_build'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Prepared Bourges data is missing columns: {sorted(missing)}')
    path = args.bourges_map
    inputs = {
        'version': BOURGES_MAP_VERSION,
        'prepared_sha256': sha256(source),
        'oligos_sha256': sha256(args.bourges_oligos),
        'hg38_sha256': sha256(args.reference),
        'chain_sha256': sha256(args.chain),
    }
    if path.is_file():
        try:
            report = json.loads(path.read_text())
            if report.get('inputs') == inputs and set(report.get('variants', {})) == set(frame.example_id):
                return
        except (OSError, ValueError, KeyError):
            pass

    raw = pd.read_csv(args.bourges_oligos, sep='\t', usecols=[
        'rsID', 'elementTag', 'allele', 'genomicSequence']).drop_duplicates()
    raw['allele'] = raw.allele.str.lower()
    positions = defaultdict(set)
    for row in frame.itertuples(index=False):
        positions[chromosome(row.chrom)].add(int(row.position) - 1)
    positions = {chrom: sorted(values) for chrom, values in positions.items()}
    anchors = defaultdict(set)
    with gzip.open(args.chain, 'rt') as handle:
        for alignment in Align.parse(handle, 'chain'):
            chrom = alignment.target.id
            if chrom not in positions:
                continue
            source_coordinates, target_coordinates = alignment.coordinates
            for a, b, c, d in zip(source_coordinates[:-1], source_coordinates[1:],
                                  target_coordinates[:-1], target_coordinates[1:]):
                if b <= a or abs(d - c) != b - a:
                    continue
                for position in positions[chrom][bisect_left(positions[chrom], a):
                                                 bisect_left(positions[chrom], b)]:
                    mapped = c + position - a if d > c else c - (position - a) - 1
                    anchors[(chrom, position)].add((alignment.query.id, int(mapped)))

    if len(reference.get('chr22', ())) != 50818468:
        raise ValueError('The supplied hg38 reference does not have GRCh38 chr22 length')
    report = {'inputs': inputs, 'variants': {}}
    for row in frame.itertuples(index=False):
        chrom = chromosome(row.chrom)
        entry = {'rsid': row.rsid, 'original_variant': {
            'chrom': chrom, 'position': int(row.position), 'ref': row.ref,
            'alt': row.alt, 'genome_build': row.genome_build}}
        try:
            candidates = anchors[(chrom, int(row.position) - 1)]
            if len(candidates) != 1:
                raise ValueError(f'Expected one lifted locus anchor; found {len(candidates)}')
            target_chrom, anchor = next(iter(candidates))
            start = max(0, anchor - SEARCH_RADIUS)
            end = min(len(reference[target_chrom]), anchor + SEARCH_RADIUS + 1)
            sequence = str(reference[target_chrom].seq[start:end]).upper()
            mapping = _match_oligos(sequence, _paired_oligos(raw, row))
            mapping['position'] += start
            entry.update(mapping, chrom=target_chrom, genome_build='GRCh38',
                         lifted_anchor_position=anchor + 1, status='ready')
        except ValueError as exc:
            entry.update(status='unresolved', error=str(exc))
        report['variants'][row.example_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def prepare_dataset(name, args, reference):
    source = args.prepared_dir / f'{name}.csv.gz'
    columns = ['example_id', 'chrom', 'position', 'ref', 'alt', 'genome_build']
    frame = pd.read_csv(source, usecols=columns, dtype={'chrom': str})[columns]
    if frame.example_id.duplicated().any():
        raise ValueError(f'{name}: duplicate example_id values.')
    frame['ref'] = frame.ref.str.upper()
    frame['alt'] = frame.alt.str.upper()
    snv = frame.ref.isin(list('ACGT')) & frame.alt.isin(list('ACGT')) & frame.ref.ne(frame.alt)
    mappings = {}
    if name == 'bourges':
        report = json.loads(args.bourges_map.read_text())
        if report['inputs']['prepared_sha256'] != sha256(source):
            raise ValueError('Bourges coordinate map does not match its prepared input.')
        if report['inputs']['hg38_sha256'] != sha256(args.reference):
            raise ValueError('Bourges coordinate map uses a different hg38 reference.')
        mappings = report['variants']
        hits = {}
    else:
        # The original Siraj analysis labels `variant` hg19, and keeps a separate
        # `variant_hg38`. Resolve only the supplied Siraj dataset's unverified
        # label using that source; never guess coordinates from allele matches.
        builds = frame.genome_build.replace('unverified', 'GRCh37') if name == 'siraj' else frame.genome_build
        if not builds.isin(['GRCh37', 'GRCh38']).all():
            raise ValueError(f'{name}: genome build must be GRCh37 or GRCh38.')
        print(f'{name}: mapping {int((snv & builds.eq("GRCh37")).sum()):,} hg19 SNVs to hg38...', flush=True)
        hits = lift_snvs(frame.loc[snv & builds.eq('GRCh37')], args.chain)
    result = []
    for i, row in enumerate(frame.itertuples(index=False)):
        out = dict(zip(columns, row))
        out.update(hg38_chrom='', hg38_position=None, hg38_ref='', hg38_alt='',
                   mapping_method='', strand='', alleles_swapped=False,
                   avi_raw_score=None, avi_phred=None, status='unsupported_non_snv')
        if snv.iat[i]:
            chrom, position, ref, alt = chromosome(row.chrom), int(row.position), row.ref, row.alt
            strand, swapped = '+', False
            if name == 'bourges':
                mapping = mappings.get(row.example_id, {})
                if mapping.get('status') != 'ready':
                    out['status'] = 'unmapped'
                    result.append(out)
                    continue
                chrom, position, ref, alt = (mapping[k] for k in ('chrom', 'position', 'ref', 'alt'))
                strand, swapped = mapping['strand'], mapping['score_multiplier'] == -1
                method = 'bourges_sequence_verified'
            elif builds.iat[i] == 'GRCh37':
                candidates = hits.get((chrom, position), set())
                if len(candidates) != 1:
                    out['status'] = 'ambiguous_mapping' if candidates else 'unmapped'
                    result.append(out)
                    continue
                chrom, position, strand = next(iter(candidates))
                if strand == '-':
                    ref, alt = ref.translate(COMPLEMENT), alt.translate(COMPLEMENT)
                method = 'hg19_to_hg38'
            else:
                method = 'input_hg38'
            out.update(hg38_chrom=chrom, hg38_position=position, hg38_ref=ref, hg38_alt=alt,
                       mapping_method=method, strand=strand, alleles_swapped=swapped)
            if chrom not in reference or not 1 <= position <= len(reference[chrom]):
                out['status'] = 'out_of_bounds'
            elif str(reference[chrom].seq[position - 1:position]).upper() != ref:
                out['status'] = 'reference_mismatch'
            else:
                out['status'] = 'ready'
        result.append(out)
    output = pd.DataFrame(result)
    output['hg38_position'] = output.hg38_position.astype('Int64')
    print(f'{name}: {len(output):,} rows; {output.status.value_counts().to_dict()}', flush=True)
    return output


def extract_scores(frame, tabix, progress, window_bp=256):
    """Fetch nearby sites together, matching POS+REF+ALT, not position alone."""
    wanted = defaultdict(dict)
    contigs = {chromosome(c): c for c in tabix.contigs}
    statuses = frame.status.tolist()
    raw_scores, phred_scores = frame.avi_raw_score.tolist(), frame.avi_phred.tolist()
    reused = 0
    for i, row in enumerate(frame.itertuples(index=False)):
        if row.status != 'ready':
            continue
        full_key = (row.hg38_chrom, int(row.hg38_position), row.hg38_ref, row.hg38_alt)
        if full_key in progress.values:
            raw_scores[i], phred_scores[i], statuses[i] = progress.values[full_key]
            reused += 1
            continue
        if row.hg38_chrom not in contigs:
            statuses[i] = 'contig_absent'
            continue
        key = (int(row.hg38_position), row.hg38_ref, row.hg38_alt)
        wanted[row.hg38_chrom].setdefault(key, []).append(i)
        statuses[i] = 'not_found'
    print(f'  Reused {reused:,} rows; {sum(map(len, wanted.values())):,} unique local lookups pending.', flush=True)
    last_update = time.monotonic()
    completed = 0
    for chrom, variants in sorted(wanted.items()):
        groups = []
        for key in sorted(variants):
            position = key[0]
            if not groups or position - groups[-1][0] >= window_bp:
                groups.append([position, position, [key]])
            else:
                groups[-1][1] = position
                groups[-1][2].append(key)
        seen = set()
        for start, end, keys in groups:
            found = {}
            # Input positions are 1-based; pysam fetch is 0-based, half-open.
            for line in tabix.fetch(contigs[chrom], start - 1, end):
                fields = line.split('\t')
                if len(fields) < 6:
                    raise ValueError('Expected AVI columns: CHROM POS REF ALT raw_score PHRED.')
                key = (int(fields[1]), fields[2], fields[3])
                if key not in variants:
                    continue
                if chromosome(fields[0]) != chrom or key in seen:
                    raise ValueError(f'Unexpected or duplicate atlas record: {fields[:4]}')
                seen.add(key)
                raw, phred = float(fields[4]), float(fields[5])
                if not (math.isfinite(raw) and math.isfinite(phred)):
                    raise ValueError(f'Nonfinite AVI score: {fields[:4]}')
                # AVI is an impact score, not a directional expression delta.
                # Never negate it for Bourges' experimental REF/ALT swap.
                found[key] = (raw, phred, 'matched')
            # Record misses only after fetch finishes successfully. An IO error
            # or interruption must not turn unprocessed variants into misses.
            for key in keys:
                value = found.get(key, (None, None, 'not_found'))
                progress.save((chrom, *key), value)
                for i in variants[key]:
                    raw_scores[i], phred_scores[i], statuses[i] = value
            if time.monotonic() - last_update > 15:
                print(f'  {chrom}:{end:,}; matched {len(seen):,} unique SNVs on this chromosome', flush=True)
                last_update = time.monotonic()
        completed += len(seen)
        progress.flush()
        print(f'  {chrom}: {len(seen):,}/{len(variants):,} unique SNVs matched ({completed:,} total)', flush=True)
    frame['avi_raw_score'], frame['avi_phred'], frame['status'] = raw_scores, phred_scores, statuses
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--atlas', type=Path, default=AVI_DIR / 'avi_scores_snvs_tabix.zip')
    parser.add_argument('--index-dir', type=Path, help='Where to save the 3 MB index extracted from the ZIP.')
    parser.add_argument('--datasets', nargs='+', choices=['ho', 'bourges', 'siraj'], default=['ho', 'bourges', 'siraj'])
    parser.add_argument('--prepared-dir', type=Path, default=WORK / 'prepared')
    parser.add_argument('--output-dir', type=Path, default=AVI_DIR)
    parser.add_argument('--chain', type=Path, default=AVI_DIR / 'hg19ToHg38.over.chain.gz')
    parser.add_argument('--reference', type=Path, default=AVI_DIR / 'hg38.2bit')
    parser.add_argument('--bourges-map', type=Path, default=AVI_DIR / 'bourges_coordinate_map.json')
    parser.add_argument('--bourges-oligos', type=Path,
                        default=ROOT / 'data/regulatory_audit/bourges_barcode_counts.txt.gz')
    parser.add_argument('--download-archive', action='store_true',
                        help='Download or resume the official AVI ZIP before extraction.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Audit coordinates and existing downloads; do not open/unpack the atlas.')
    args = parser.parse_args(argv)
    tabix = None
    progress = None
    atlas = index = None
    try:
        if args.download_archive:
            download_archive(args.atlas)
        if not args.dry_run:
            atlas, index = resolve_atlas(args.atlas, args.index_dir)
            if atlas.suffix == '.zip':
                tabix = ZipTabix(atlas, index)
            else:
                try:
                    import pysam
                except ImportError as exc:
                    raise ValueError('Install the reader: venv/bin/python -m pip install pysam') from exc
                tabix = pysam.TabixFile(str(atlas), index=str(index))
            expected_header = ['CHROM', 'POS', 'REF', 'ALT', 'raw_score', 'PHRED']
            if not any(line.lstrip('#').split('\t')[:6] == expected_header for line in tabix.header):
                raise ValueError('The indexed file is not the expected AVI raw/PHRED score table.')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # A lock avoids two runs overwriting the same output files.
        with (args.output_dir / '.extract.lock').open('w') as lock, args.reference.open('rb') as handle:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if tabix is not None:
                progress = LocalProgress(args.output_dir / 'extraction_progress.sqlite3', atlas, index)
            records = SeqIO.parse(handle, 'twobit')
            reference = {name: records[name] for name in records.keys()}
            if {'bourges', 'siraj'} & set(args.datasets):
                ensure_chain(args.chain)
            if 'bourges' in args.datasets:
                prepare_bourges_map(args, reference)
            for name in dict.fromkeys(args.datasets):
                frame = prepare_dataset(name, args, reference)
                if tabix is not None:
                    extract_scores(frame, tabix, progress)
                suffix = 'coordinates' if args.dry_run else 'avi'
                destination = args.output_dir / f'{name}.{suffix}.csv.gz'
                temporary = destination.with_name(destination.name + '.partial')
                frame.to_csv(temporary, index=False, compression='gzip')
                temporary.replace(destination)
                summary = {
                    'dataset': name, 'rows': len(frame), 'status_counts': frame.status.value_counts().to_dict(),
                    'dry_run': args.dry_run, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'input_sha256': sha256(args.prepared_dir / f'{name}.csv.gz'),
                    'atlas': str(atlas) if atlas else None, 'index': str(index) if index else None,
                    'atlas_size_bytes': atlas.stat().st_size if atlas else None,
                    'atlas_mtime_ns': atlas.stat().st_mtime_ns if atlas else None,
                    'atlas_genome_build': 'GRCh38', 'siraj_build_source': SIRAJ_BUILD_SOURCE if name == 'siraj' else None,
                    'score_meaning': 'AVI raw and PHRED impact scores; not cell-type track deltas',
                }
                (args.output_dir / f'{name}.{suffix}.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                print(f'Saved {destination}: {summary["status_counts"]}', flush=True)
        return 0
    except KeyboardInterrupt:
        print('\nStopped. Completed local lookups are saved; rerun to resume.', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, sqlite3.Error, zipfile.BadZipFile) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        if progress is not None:
            progress.close()
        if tabix is not None:
            tabix.close()


if __name__ == '__main__':
    sys.exit(main())
