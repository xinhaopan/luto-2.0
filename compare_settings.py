"""Compare model settings between OLD (FLC_medium) and NEW (flat FLC) AgS3 runs."""
import os

OLD_SET = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__00_29_59_RF5_2010-2050/model_run_settings.txt'
NEW_BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_3_SCN_AgS3/output'
subdirs = [d for d in os.listdir(NEW_BASE) if os.path.isdir(os.path.join(NEW_BASE, d))]
NEW_SET = os.path.join(NEW_BASE, subdirs[0], 'model_run_settings.txt')

print('=== Settings comparison OLD vs NEW AgS3 ===')
with open(OLD_SET) as f:
    old_lines = {line.split(':')[0].strip(): line.strip() for line in f if ':' in line}
with open(NEW_SET) as f:
    new_lines = {line.split(':')[0].strip(): line.strip() for line in f if ':' in line}

# Show all keys where OLD != NEW
print('\nDIFFERENT settings:')
all_keys = sorted(set(list(old_lines.keys()) + list(new_lines.keys())))
for k in all_keys:
    o = old_lines.get(k, '<MISSING>')
    n = new_lines.get(k, '<MISSING>')
    if o != n:
        print(f'  KEY: {k}')
        print(f'    OLD: {o[:120]}')
        print(f'    NEW: {n[:120]}')

# Show FLC/AC related settings
print('\n\nFLC/AC related settings in OLD:')
for k, v in old_lines.items():
    if any(x in k.upper() for x in ['FLC', 'AC', 'AREA_COST', 'PRODUCTIVITY', 'DEMAND']):
        print(f'  {v[:120]}')
