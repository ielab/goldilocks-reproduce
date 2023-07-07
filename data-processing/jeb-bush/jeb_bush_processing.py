import re


def athome1_rel_load(dataset:str):
    """ athome dataset from 100 to 109. """
    jb_rel_dict = {}
    path = f"./rels/rel.judgments_athome{dataset}"
    with open(path, 'r')as f:
        lines = f.readlines()[1:]
        doc_id_list = []
        for line in lines:
            doc_id = line.split('-')[1].strip('\n')
            if len(doc_id) < 6:
                doc_id = "0" * (6-len(doc_id)) + doc_id
            if doc_id not in clean_docs: # replace duplicate
                for k in dul_map.keys():
                    if doc_id in dul_map[k]:
                        doc_id = k
                    else:
                        continue
            doc_id_list.append(doc_id)
        jb_rel_dict[dataset] = doc_id_list
    return 


def get_raw(path:str):
    """get concatenated email from subject and body."""
    
    with open(path, 'r', encoding="latin1") as f:
        mail = {'subject':'', 'body':''}
        lines = f.readlines()
        content = [re.sub(r'\t|>|-|_|\*|<|#|@|\'','',line.strip())+' ' for line in lines if line!='\n']
        head = [(i,line.strip()) for (i, line) in enumerate(content) if re.match(r'^Subject:|^Re:', line)!=None]
        for h in head:
            if 'Subject' in h[1]:
                mail['subject'] = h[1][8:].strip()
            elif 'Re' in h[1] and mail['subject']==[]:
                mail['subject'] = h[1][3:].strip()
            else: 
                pass
        text = ''
        body = text.join(content).strip()
        body = re.sub(r'\s{2}', '', body)
        mail['body'] = body
    raw = mail['subject'] + ' ' + mail['body']
    return raw


def test(path= f"./athome4.facetsandqrels"):
    """ athome dataset from 401 to 434. """
    jb_rel_dict = {}
    with open(path, 'r')as f:
        lines = f.readlines()
        for line in lines:
            dataset, doc_id, rel , _ = line.split(' ')
            if doc_id not in clean_docs: # replace duplicate
                    for k in dul_map.keys():
                        if doc_id in dul_map[k]:
                            doc_id = k
                        else:
                            continue
            if dataset not in jb_rel_dict.keys():
                jb_rel_dict[dataset] = [] # [] > {}
                jb_rel_dict[dataset].append((doc_id.strip('\n'), rel)) # tuple or dict
            else:
                jb_rel_dict[dataset].append((doc_id.strip('\n'), rel))
    return jb_rel_dict