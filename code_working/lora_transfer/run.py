import argparse
from downstream_utility.run_v2 import install_guard,FORBIDDEN
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','cache_public','cache_private','e4_replay','train_public','train_pooled','generate','classifier_replay','classify','evaluate','verify'])
    args=parser.parse_args();install_guard()
    if args.phase=='verify':
        from .verify import verify
        verify()
    else:
        from . import engine
        engine.dispatch(args.phase)
    assert not FORBIDDEN.intersection(sys.modules)

if __name__=='__main__':main()
