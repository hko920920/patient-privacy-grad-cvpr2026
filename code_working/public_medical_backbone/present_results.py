"""Present immutable generated PNGs after both blind reviews are committed."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parents[1]/'_reports/public_medical_backbone_20260916_v1'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def render():
    for phase in ('selection','confirmation'):
        decision_path=OUT/(phase+'_decision.json')
        if not decision_path.exists():continue
        decision=read(decision_path.name);manifest=read('manifest_'+phase+'.json')
        for checkpoint in sorted(decision['checkpoint_results']):
            records=[r for r in manifest if r['phase']==phase and r['checkpoint']==checkpoint]
            prompt_order={'generic':0,'normal':1,'effusion':2,'cardiomegaly':3}
            records.sort(key=lambda r:(prompt_order[r['task_id'].split('_')[1]],int(r['task_id'].split('_')[2])))
            canvas=Image.new('RGB',(1024,1152),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f'{phase} {checkpoint} | two assistant reviews: {decision["checkpoint_results"][checkpoint]["total_pass"]}/16 | morphology only',fill='black')
            for index,row in enumerate(records):
                x=(index%4)*256;y=32+(index//4)*280
                good=decision['joint_pass_by_image'][row['image_id']]
                caption=row['task_id'].replace(phase+'_','')+' | '+('PASS' if good else 'FAIL')
                draw.text((x+5,y+5),caption,fill='darkgreen' if good else 'darkred')
                with Image.open(OUT/row['image_path']) as source:canvas.paste(source,(x,y+24))
            target=OUT/(phase+'_'+checkpoint+'_reviewed.png')
            if target.exists():raise RuntimeError('Preserve existing figure '+str(target))
            canvas.save(target)
    records=[r for r in read('manifest_selection.json') if r['phase']=='diagnostic']
    records.sort(key=lambda r:(r['checkpoint'],r['task_id']))
    canvas=Image.new('RGB',(1024,280),'white');draw=ImageDraw.Draw(canvas)
    for index,row in enumerate(records):
        draw.text((index*256+5,5),row['checkpoint']+' '+row['task_id'],fill='black')
        with Image.open(OUT/row['image_path']) as source:canvas.paste(source,(index*256,24))
    target=OUT/'diagnostic_four.png'
    if target.exists():raise RuntimeError('Preserve existing diagnostic figure')
    canvas.save(target)
if __name__=='__main__':render()
