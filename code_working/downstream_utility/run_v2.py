"""Safe corrected entrypoint. Legacy data/train imports fail before execution."""
import argparse,importlib.abc,sys

FORBIDDEN={'downstream_utility.data','downstream_utility.train'}
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in FORBIDDEN:
            raise RuntimeError('Forbidden legacy classifier module: '+fullname)
        return None

def install_guard():
    if FORBIDDEN & set(sys.modules):raise RuntimeError('Legacy module already loaded')
    if not any(isinstance(x,RejectLegacy) for x in sys.meta_path):sys.meta_path.insert(0,RejectLegacy())

def main():
    p=argparse.ArgumentParser(description='Corrected all-arm one-step integration replay; no generation or efficacy evaluation.')
    p.add_argument('phase',choices=['prepare','classifier_corrected','verify_corrected'])
    args=p.parse_args();install_guard()
    if args.phase=='verify_corrected':
        from .verify_all_arm_replay import verify
        verify()
    else:
        from .all_arm_replay import prepare,execute
        (prepare if args.phase=='prepare' else execute)()
    if FORBIDDEN & set(sys.modules):raise RuntimeError('Legacy module imported during corrected execution')

if __name__=='__main__':main()
