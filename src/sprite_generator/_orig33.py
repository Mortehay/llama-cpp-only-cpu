import os, psycopg2
from PIL import Image, ImageDraw, ImageFont
conn=psycopg2.connect(os.environ['DB_URL']); cur=conn.cursor()
cur.execute("SELECT file_path FROM reference_assets WHERE kind='tile' AND deleted=false "
            "AND trainable=true AND coalesce(metrics->>'parent_id','')='' ORDER BY created_at")
paths=[r[0] for r in cur.fetchall()]
CELL=170; cols=7
rows=(len(paths)+cols-1)//cols
sheet=Image.new("RGB",(cols*CELL, rows*CELL+26),(24,24,38))
d=ImageDraw.Draw(sheet)
try: f=ImageFont.load_default(size=16)
except TypeError: f=ImageFont.load_default()
d.text((6,6), f"the {len(paths)} single-subject ORIGINALS - the sharp 8% of the tile set", font=f, fill=(226,224,240))
for i,p in enumerate(paths):
    try:
        im=Image.open(p).convert("RGBA"); im.thumbnail((CELL-8,CELL-8), Image.LANCZOS)
        bg=Image.new("RGB", im.size,(16,16,26)); bg.paste(im,(0,0),im)
        sheet.paste(bg,( (i%cols)*CELL+(CELL-im.width)//2, 26+(i//cols)*CELL+(CELL-im.height)//2))
    except Exception: pass
sheet.save("/app/images/_orig33.png")
print("saved", len(paths))
