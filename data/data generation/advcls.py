import json


jsons = [json.load(open(rf"C:\Users\inbaryeh\Desktop\ann\sent_{i:05}.json")) for i in range(68124)]
found = {'subj': [], 'obj': [], 'none': []}
for j in jsons:
    text = j['text']
    label = j['gold_relations'][0]['labels'][0]
    if label != 'no_relation':
        arg1 = j['gold_relations'][0]['arguments']['subject'][0]['tokenInterval']
        arg1_range = range(arg1['start'], arg1['end'])
        arg2 = j['gold_relations'][0]['arguments']['object'][0]['tokenInterval']
        arg2_range = range(arg2['start'], arg2['end'])
    edges = j['sentences'][0]['graphs']['universal-enhanced']['edges']
    for edge in edges:
        if 'advcl' in edge['relation']:
            source_subjs = [c['destination'] for c in edges if (c['source'] == edge['source']) and ("subj" in c['relation'])]
            if not source_subjs:
                continue
            source_objs = [c['destination'] for c in edges if (c['source'] == edge['source']) and ("obj" in c['relation'])]
            if not source_objs:
                continue
            if any(['subj' in c['relation'] for c in edges if c['source'] == edge['destination']]):
                continue
            dest_childs_and_self = [c['destination'] for c in edges if c['source'] == edge['destination']] + [edge['destination']]
            
            has_found = (False, None)
            if label != 'no_relation':
                if any(c in arg1_range for c in dest_childs_and_self):
                    cur_subj = [c for c in source_subjs if c in arg2_range]
                    cur_obj = [c for c in source_objs if c in arg2_range]
                    store = (text, label, cur_subj, cur_obj, edge)
                    if cur_subj:
                        has_found = (True, store)
                        found['subj'].append(store)
                    elif cur_obj:
                        has_found = (True, store)
                        found['obj'].append(store)
                elif any(c in arg2_range for c in dest_childs_and_self):
                    cur_subj = [c for c in source_subjs if c in arg1_range]
                    cur_obj = [c for c in source_objs if c in arg1_range]
                    store = (text, label, cur_subj, cur_obj, edge)
                    if cur_subj:
                        has_found = (True, store)
                        found['subj'].append(store)
                    elif cur_obj:
                        has_found = (True, store)
                        found['obj'].append(store)
            
            if has_found[0]:
                print(has_found[1])
            else:
                found['none'].append((text, label, source_subjs, source_objs, edge))


len(found['none'])
len(found['obj'])
len(found['subj'])


examples = dict()
def store(i, j, subj_i=0, obj_i=0, ignore=False):
    if (not ignore) and (found['none'][i][0] in examples):
        print("already found text")
        return examples[found['none'][i][0]]
    seq = ['O'] * len(found['none'][i][0].split())
    seq2 = ['O'] * len(found['none'][i][0].split())
    if j == 0 or j == 2:
        seq[found['none'][i][2][subj_i]] = 'S'
        seq2[found['none'][i][2][subj_i]] = 'S'
    if j == 1 or j == 2:
        seq[found['none'][i][3][obj_i]] = 'S'
        seq2[found['none'][i][3][obj_i]] = 'S'
    seq2[found['none'][i][4]['destination']] = 'P'
    examples[found['none'][i][0]] = {'text':found['none'][i][0], 'class':j, 'seq':seq, 'seq2':seq2}


i = 1
def nice_print(k):
    for ss in found['none'][k][2]:
        for oo in found['none'][k][3]:
            if ss < found['none'][k][4]['source'] < oo < found['none'][k][4]['destination']:
                print(" ".join(found['none'][k][0].split()[:ss]) + " " + Back.YELLOW + Fore.BLACK + found['none'][k][0].split()[ss] + Style.RESET_ALL + " " + " ".join(found['none'][k][0].split()[ss + 1:found['none'][k][4]['source']]) + " " + Back.GREEN + Fore.BLACK + found['none'][k][0].split()[found['none'][k][4]['source']] + Style.RESET_ALL + " " + " ".join(found['none'][k][0].split()[found['none'][k][4]['source'] + 1:oo]) + " " + Back.YELLOW + Fore.BLACK + found['none'][k][0].split()[oo] + Style.RESET_ALL + " " + " ".join(found['none'][k][0].split()[oo + 1:found['none'][k][4]['destination']]) + " " + Back.GREEN + Fore.BLACK + found['none'][k][0].split()[found['none'][k][4]['destination']] + Style.RESET_ALL + " " + " ".join(found['none'][k][0].split()[found['none'][k][4]['destination'] + 1:]))
            else:
                print(found['none'][k][0] + "\nsubj:" + found['none'][k][0].split()[ss] + "\nadvcl-father:" + found['none'][k][0].split()[found['none'][k][4]['source']] + "\nobj:" + found['none'][k][0].split()[oo] + "\nadvcl-child:" + found['none'][k][0].split()[found['none'][k][4]['destination']])



# both?: When the Mohammed cartoons -- published in September 2005 by the Danish newspaper Jyllands-Posten to defy rising self-censorship after van Gogh 's murder -- were answered by worldwide violence , only one major American (S:) newspaper , the Philadelphia Inquirer , (V1:) joined such European (O:) dailies as Die Welt and El País in (V2:) reprinting them as a gesture of free-speech solidarity .
# Neither?: Actress Mia (S:) Farrow has arrived in Cambodia and plans to (V1:) defy a (O:) ban on (V2:) holding a ceremony at a former Khmer Rouge prison , as part of her campaign on Darfur , activists said Saturday .
