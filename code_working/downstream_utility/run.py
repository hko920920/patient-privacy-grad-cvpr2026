import argparse
from .common import OUT

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','data','generate','classifier','verify']);a=p.parse_args()
    if a.phase=='prepare':
        from .prepare import prepare
        prepare()
    elif a.phase=='data':
        from .data import prepare_cache
        prepare_cache()
    elif a.phase=='generate':
        from .generate import generate
        generate()
    elif a.phase=='classifier':
        from .train import profile_classifier
        profile_classifier()
    else:
        from .verify import verify
        verify()
if __name__=='__main__':main()
