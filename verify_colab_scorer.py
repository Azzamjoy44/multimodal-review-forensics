"""
verify_colab_scorer.py — pre-flight: confirm the Colab notebook's generic score_dir() exactly
reproduces the project's canonical serving paths on CPU, BEFORE spending a Colab run.
RUN: python verify_colab_scorer.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import onnxruntime as ort
import model_b
from score_reviews import _run_yelp_lstm, _run_yelp_distilbert

FEAT = model_b.FEAT_COLS
N = 200


class KerasTok:
    def __init__(s, wi, filt, lower, nw): s.wi=wi; s._f=str.maketrans('','',filt); s.lower=lower; s.nw=nw
    @classmethod
    def load(cls, p):
        cfg=json.load(open(p,encoding='utf-8'))['config']; wi=cfg['word_index']
        if isinstance(wi,str): wi=json.loads(wi)
        return cls(wi, cfg.get('filters','!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'), cfg.get('lower',True), cfg.get('num_words'))
    def seqs(s, texts):
        out=[]
        for t in texts:
            if s.lower: t=t.lower()
            t=t.translate(s._f); seq=[]
            for w in t.split():
                i=s.wi.get(w)
                if i is not None and (s.nw is None or i<s.nw): seq.append(i)
            out.append(seq)
        return out


def pad(seqs, maxlen):
    a=np.zeros((len(seqs),maxlen),dtype=np.float32)
    for i,s in enumerate(seqs):
        s=s[:maxlen]; a[i,:len(s)]=s
    return a


def read_thr(d):
    for jf in glob.glob(os.path.join(d,'*.json')):
        try: o=json.load(open(jf))
        except Exception: continue
        if isinstance(o,dict) and 'threshold' in o: return float(o['threshold'])
    return 0.5


def score_dir(d, texts, beh_raw):
    sess=ort.InferenceSession(glob.glob(os.path.join(d,'*.onnx'))[0], providers=['CPUExecutionProvider'])
    inames=[i.name for i in sess.get_inputs()]; thr=read_thr(d)
    def mlen(name, dflt):
        sh=[i.shape[1] for i in sess.get_inputs() if i.name==name][0]
        return sh if isinstance(sh,int) else dflt
    if 'input_layer' in inames:
        tok=KerasTok.load(glob.glob(os.path.join(d,'*_tokenizer.json'))[0])
        probs=sess.run(None,{'input_layer':pad(tok.seqs(texts), mlen('input_layer',150))})[0].reshape(-1)
    elif 'text_input' in inames:
        import joblib; sc=joblib.load(glob.glob(os.path.join(d,'*_scaler.joblib'))[0])
        tok=KerasTok.load(glob.glob(os.path.join(d,'*_tokenizer.json'))[0])
        beh=sc.transform(beh_raw).astype(np.float32)
        probs=sess.run(None,{'text_input':pad(tok.seqs(texts), mlen('text_input',250)),'beh_input':beh})[0].reshape(-1)
    else:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(d); is_mb=('beh' in inames); beh=None
        if is_mb:
            import joblib; sc=joblib.load(glob.glob(os.path.join(d,'*_scaler.joblib'))[0]); beh=sc.transform(beh_raw).astype(np.float32)
        out=[]; B=64
        for s in range(0,len(texts),B):
            enc=tok(texts[s:s+B],truncation=True,padding='max_length',max_length=256,return_tensors='np')
            feed={'input_ids':enc['input_ids'].astype('int64'),'attention_mask':enc['attention_mask'].astype('int64')}
            if is_mb: feed['beh']=beh[s:s+B]
            r=np.asarray(sess.run(None,feed)[0],dtype=np.float64)
            if r.ndim==2 and r.shape[1]==2:
                e=np.exp(r-r.max(1,keepdims=True)); p=(e/e.sum(1,keepdims=True))[:,1]
            else:
                p=1/(1+np.exp(-r.reshape(-1)))
            out.append(p)
        probs=np.concatenate(out)
    return (probs>=thr).astype(int)


def main():
    df = pd.read_csv("data/yelp_frontend_multimodal.csv").head(N)
    df["review_text"] = df["review_text"].astype(str)
    texts = df["review_text"].tolist()
    beh = df[FEAT].astype("float32").values
    reviews = [{"review_text": texts[i], **df.iloc[i][FEAT].to_dict()} for i in range(len(df))]

    def canon(labels): return np.array([(-1 if (d or {}).get("label") is None else d["label"]) for d in labels])

    cases = []
    # baseline
    cases.append(("base_lstm", score_dir("data/yelp_fake_lstm_onnx", texts, beh), canon(_run_yelp_lstm(texts, "baseline"))))
    cases.append(("base_db",   score_dir("data/yelp_fake_distilbert_onnx", texts, beh), canon(_run_yelp_distilbert(texts, "baseline"))))
    # OLD MB (default dirs)
    model_b._lstm=None; model_b._db=None
    cases.append(("old_lstm", score_dir("data/yelp_multimodal_lstm_onnx", texts, beh), canon(model_b._score_lstm(reviews))))
    cases.append(("old_db",   score_dir("data/yelp_multimodal_distilbert_onnx", texts, beh), canon(model_b._score_db(reviews))))
    # NEW MB (full dirs)
    model_b.LSTM_DIR="data/yelp_multimodal_lstm_full_onnx"; model_b.DB_DIR="data/yelp_multimodal_distilbert_full_onnx"
    model_b._lstm=None; model_b._db=None
    cases.append(("new_lstm", score_dir("data/yelp_multimodal_lstm_full_onnx", texts, beh), canon(model_b._score_lstm(reviews))))
    cases.append(("new_db",   score_dir("data/yelp_multimodal_distilbert_full_onnx", texts, beh), canon(model_b._score_db(reviews))))

    print(f"\nscore_dir (notebook) vs canonical serving path — agreement on {N} reviews:")
    allok = True
    for name, a, b in cases:
        agree = (a == b).mean()
        allok &= (agree == 1.0)
        print(f"  {name:<10} {agree:6.1%}  [{'OK' if agree==1.0 else 'MISMATCH'}]")
    print("\n", "ALL MATCH — notebook scorer is correct, safe to run on Colab." if allok
          else "MISMATCH — fix the notebook before the Colab run.")


if __name__ == "__main__":
    main()
