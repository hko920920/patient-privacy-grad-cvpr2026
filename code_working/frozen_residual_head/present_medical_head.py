"""Export all paired originals as labeled contact sheets for the review report."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from .run_medical_head import OUT
from public_medical_backbone.run_pilot import RESEARCH,require,read,save,sha

def main(out):
    c=read(out/'contract.json');m=read(out/'generation_manifest.json');review=read(out/'visual_review_root.json')
    require(review['images_reviewed']==192 and review['method_labels_hidden'],'Record full hidden-label review before labeled presentation')
    require(read(out/'generation_verification.json')['complete'],'Numerical verification first')
    dest=RESEARCH/'figures/medical_head_20260916';dest.mkdir(parents=True,exist_ok=False)
    index={(x['seed_id'],x['prompt_id'],x['method']):x for x in m};cells=[(sid,pid) for sid in c['generation_seeds'] for pid in c['prompts']]
    exported=[]
    for offset in range(0,len(cells),8):
        grid=Image.new('RGB',(768,2272),'white');draw=ImageDraw.Draw(grid)
        for col,name in enumerate(['BACKBONE','PUBLIC HEAD','POOLED HEAD']):draw.text((col*256+8,8),name,fill='black')
        for row,(sid,pid) in enumerate(cells[offset:offset+8]):
            for col,method in enumerate(c['generation_methods']):
                record=index[(sid,pid,method)];path=out/record['image_path'];require(sha(path)==record['image_sha256'],'Original PNG binding')
                x=col*256;y=32+row*280;draw.text((x+4,y+4),sid+' '+pid,fill='black')
                with Image.open(path) as im:grid.paste(im,(x,y+24))
        path=dest/f'paired_{offset//8+1}.png';grid.save(path)
        exported.append(dict(path=path.relative_to(RESEARCH).as_posix(),sha256=sha(path),cells=cells[offset:offset+8]))
    save(out/'presentation.json',dict(all_images=192,methods=c['generation_methods'],image_pixels_unmodified=True,exported=exported))
    print('EXPORTED_ALL192_PAIRED_IMAGES',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();main(a.out)
