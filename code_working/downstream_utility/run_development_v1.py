"""Prospective non-DP development only; no final, reserved, or DP phases."""
import argparse
from .run_v2 import install_guard, FORBIDDEN
import sys

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('phase', choices=['prepare_contract','preflight','generate_512','calibrate_R1','train_21_runs','evaluate_method_development','verify_all','execute'])
    args=p.parse_args();install_guard()
    from . import development_v1 as d
    if args.phase=='execute':
        for phase in ['preflight','generate_512','calibrate_R1','train_21_runs','evaluate_method_development','verify_all']:
            getattr(d,phase)()
    else:getattr(d,args.phase)()
    if FORBIDDEN & set(sys.modules):raise RuntimeError('Legacy module imported')

if __name__=='__main__':main()
