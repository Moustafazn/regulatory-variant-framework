"""Download public study inputs with integrity checks, or verify an existing data directory.

No credentials or private experiment outputs are required. Downloads are explicit.
"""
import argparse
import gzip
import hashlib
import json
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from regulatory_model.reporting import sha256,write_json

ASSETS={
 'ho':('regulatory_audit/ho_supp.xlsx',
       'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-025-02301-3/MediaObjects/41588_2025_2301_MOESM4_ESM.xlsx',
       'f184e03c2a42d1673a05c5660edaee2e67d603be3b02d6f003ef8287c55322b5'),
 'replicates':('regulatory_audit/ho_normalized_counts.txt.gz',
       'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297471/suppl/GSE297471_OLJR.C_20240310_Tcell_normalized_counts.out.txt.gz',
       '29cdd5d8ca628aa8247b362351a61dc2bd8ca93939321916356d7bf8840f79a7'),
 'chain':('reference/hg19ToHg38.over.chain.gz',
       'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz',
       '5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d'),
 'jurkat':('accessibility/jurkat_ENCFF916JRN.bigWig',
       'https://www.encodeproject.org/files/ENCFF916JRN/@@download/ENCFF916JRN.bigWig',
       'faf09ebafbc68d9f89139099339c2860ffd7e04236a73affca6a7879c9920e12'),
 'cd4':('accessibility/cd4_ENCFF124RFI.bigWig',
       'https://www.encodeproject.org/files/ENCFF124RFI/@@download/ENCFF124RFI.bigWig',
       '23a6e49cb103bf4f41a33a0aa52d1a33d2ed5fdb88ae20737b6177ed4fc828a7'),
 'reference':('reference/hg38.fa.gz',
       'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',None)
}


def download(url,destination,expected=None):
    import certifi
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.part')
    request=urllib.request.Request(url,headers={'User-Agent':'regulatory-variant-framework/2.0 public-data research'})
    context=ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request,context=context,timeout=120) as source,temporary.open('wb') as target:
            shutil.copyfileobj(source,target,1024*1024)
        if expected and sha256(temporary)!=expected:
            raise ValueError(f'Checksum mismatch for {destination.name}; downloaded content was not accepted.')
        temporary.replace(destination)
    finally:
        if temporary.exists():temporary.unlink()


def unpack_reference(archive):
    """Bind the decompressed FASTA to the verified archive, including existing files."""
    fasta=archive.with_suffix('');part=fasta.with_name(fasta.name+'.part')
    expected=hashlib.sha256()
    try:
        if fasta.exists():
            with gzip.open(archive,'rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):expected.update(block)
            if sha256(fasta)!=expected.hexdigest():
                raise ValueError(f'{fasta}: contents differ from the verified archive; existing file left untouched.')
        else:
            with gzip.open(archive,'rb') as source,part.open('wb') as target:
                for block in iter(lambda:source.read(1024*1024),b''):
                    expected.update(block);target.write(block)
            part.replace(fasta)
    finally:
        if part.exists():part.unlink()
    return expected.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--download',action='store_true',help='Fetch missing assets (~2 GB compressed; allow 8 GB disk).')
    parser.add_argument('--asset',choices=list(ASSETS),action='append',help='Restrict setup/verification to named assets.')
    args=parser.parse_args(argv);records={}
    for name in args.asset or ASSETS:
        relative,url,expected=ASSETS[name];path=args.data_dir/relative
        if not path.exists():
            if not args.download:raise FileNotFoundError(f'{path}; rerun with --download or provide public files manually.')
            print(f'Downloading {name} from {url}',flush=True);download(url,path,expected)
        actual=sha256(path)
        if expected and actual!=expected:raise ValueError(f'{path}: SHA-256 mismatch; existing file left untouched.')
        record={'url':url,'path':str(path),'sha256':actual,'expected_sha256':expected,'bytes':path.stat().st_size}
        if name=='reference':
            # UCSC supplies an authoritative checksum for the compressed genome.
            checksum=args.data_dir/'reference'/'ucsc_md5sum.txt'
            if not checksum.exists():
                if not args.download:raise FileNotFoundError('UCSC checksum manifest missing; run with --download.')
                download('https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/md5sum.txt',checksum)
            expected_md5=next(line.split()[0] for line in checksum.read_text().splitlines() if line.split()[-1].lstrip('*')=='hg38.fa.gz')
            h=hashlib.md5()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=expected_md5:raise ValueError('UCSC FASTA compressed-file MD5 mismatch.')
            record['fasta_sha256']=unpack_reference(path);record['ucsc_md5']=expected_md5
        records[name]=record
        print(f'Verified {name}: {actual}',flush=True)
    manifest=args.data_dir/'public_input_manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    write_json(manifest,{**previous,**records})
    return records

if __name__=='__main__':main()
