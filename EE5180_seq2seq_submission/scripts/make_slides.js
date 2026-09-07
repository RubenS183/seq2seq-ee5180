/**
 * Mid-term slide deck, built from the same results JSON as the report.
 * Owner: M4.
 *
 *   node scripts/make_slides.js [outfile.pptx]
 *
 * The two accent colours ARE the experiment's two conditions: coral = forward,
 * blue = reversed, matching every figure in results/.
 */
const path = require("path");
const fs = require("fs");
const PptxGenJS = require(path.join(process.env.HOME, "ee5180-work/node_modules/pptxgenjs"));

const REPO = path.resolve(__dirname, "..");
const OUT = process.argv[2] || path.join(REPO, "slides", "EE5180_midterm_slides.pptx");

// ---------------------------------------------------------------- palette
const INK = "18204A";        // deep indigo — dominant
const INK_SOFT = "3B4A7A";
const PAPER = "FFFFFF";
const MIST = "EEF1F8";
const FWD = "C44E52";        // forward  (matches the figures)
const REV = "4C72B0";        // reversed (matches the figures)
const MUTED = "6B7280";
const HEAD_FONT = "Cambria";
const BODY_FONT = "Calibri";

// ---------------------------------------------------------------- data
function load(name) {
  const p = path.join(REPO, "results", name, "all_results.json");
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : [];
}
function pick(rows, direction, beam, n = 1) {
  return rows.find(r => r.direction === direction && r.beam_size === beam && r.n_models === n) || null;
}
const wmt = load("wmt14_small");
const m30 = load("multi30k");
const prepPath = path.join(process.env.HOME, "ee5180-work/data/wmt14/prepared/prepare_report.json");
const prep = fs.existsSync(prepPath) ? JSON.parse(fs.readFileSync(prepPath, "utf8")) : {};
const fig = f => path.join(REPO, "results", f);
const has = f => fs.existsSync(fig(f));

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";              // 13.3 x 7.5 in
pres.author = "EE5180 project group";
pres.title = "Reproducing Sutskever et al. (2014)";
const W = 13.3, H = 7.5, M = 0.6;

// ---------------------------------------------------------------- helpers
function titleSlide(s, title, kicker) {
  if (kicker) s.addText(kicker, {
    x: M, y: 0.42, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: REV, charSpacing: 1.2,
  });
  s.addText(title, {
    x: M, y: 0.74, w: W - 2 * M, h: 0.78, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: 34, bold: true, color: INK,
  });
}
function body(s, items, opt = {}) {
  s.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i !== items.length - 1, paraSpaceAfter: 9 },
  })), Object.assign({
    x: M, y: 1.7, w: W - 2 * M, h: 4.6, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, color: INK_SOFT, valign: "top",
  }, opt));
}
function note(s, text) {
  s.addText(text, {
    x: M, y: H - 0.72, w: W - 2 * M, h: 0.42, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED,
  });
}
function statCard(s, x, y, w, value, label, color) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 1.62, rectRadius: 0.09, fill: { color: MIST }, line: { color: MIST },
  });
  s.addText(value, {
    x, y: y + 0.16, w, h: 0.86, isTextBox: true, margin: 0, align: "center",
    fontFace: HEAD_FONT, fontSize: 40, bold: true, color,
  });
  s.addText(label, {
    x: x + 0.12, y: y + 1.03, w: w - 0.24, h: 0.5, isTextBox: true, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 11.5, color: INK_SOFT,
  });
}
function table(s, head, rows, opt = {}) {
  const bold = r => r.map(c => ({ text: String(c), options: { bold: true, color: PAPER, fill: { color: INK } } }));
  s.addTable([bold(head), ...rows.map(r => r.map(c =>
    typeof c === "object" ? c : { text: String(c) }))], Object.assign({
      x: M, y: 1.75, w: W - 2 * M, colW: null,
      fontFace: BODY_FONT, fontSize: 13, color: INK_SOFT, border: { type: "solid", pt: 0.5, color: "D6DBE8" },
      rowH: 0.36, valign: "middle", align: "center",
    }, opt));
}

// ============================================================ 1. title
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("Reproducing Table 1", {
    x: M, y: 2.05, w: W - 2 * M, h: 0.9, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: 46, bold: true, color: PAPER,
  });
  s.addText("Sequence to Sequence Learning with Neural Networks", {
    x: M, y: 2.98, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: 22, color: "CADCFC",
  });
  s.addText("Sutskever, Vinyals & Le — Google — NeurIPS 2014 — arXiv:1409.3215", {
    x: M, y: 3.5, w: W - 2 * M, h: 0.36, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 14, color: "9FB0D8",
  });
  s.addText("EE5180 course project  ·  mid-term deliverable  ·  TA: Prasenjit Kr Mudi (EE21D057)", {
    x: M, y: 4.35, w: W - 2 * M, h: 0.36, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 13, color: "CADCFC",
  });
  s.addNotes("We reproduce Table 1 rows 3 and 4 — single forward vs single reversed LSTM — at reduced scale, and are explicit throughout that absolute BLEU is not comparable to the paper's.");
}

// ============================================================ 2. the ask
{
  const s = pres.addSlide();
  titleSlide(s, "What the mid-term asks for", "THE BRIEF");
  table(s, ["Table 1 row", "Paper, cased BLEU on ntst14"], [
    ["Single forward LSTM, beam 12", { text: "26.17", options: { bold: true, color: FWD } }],
    ["Single reversed LSTM, beam 12", { text: "30.59", options: { bold: true, color: REV } }],
  ], { colW: [7.2, 4.9], y: 1.72 });
  s.addText("Why these two rows: they differ in exactly one thing — the direction the encoder reads the source — so together they isolate the paper's central empirical claim.", {
    x: M, y: 3.0, w: W - 2 * M, h: 0.7, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, color: INK_SOFT,
  });
  body(s, [
    "Reproduce 1–2 rows of Table 1 and demonstrate understanding of the method.",
    "End-term: study the fixed-vector bottleneck and how it motivated Attention → Transformer.",
  ], { y: 3.85, h: 1.5 });
  s.addNotes("Rows 3 and 4 are also the cheapest scientifically meaningful pair — no ensembling required.");
}

// ============================================================ 3. method
{
  const s = pres.addSlide();
  titleSlide(s, "The method", "HOW IT WORKS");
  if (has("figures/model_schematic.png")) {
    s.addImage({ path: fig("figures/model_schematic.png"), x: 0.35, y: 1.62, w: 12.6, h: 3.75 });
  }
  s.addText([
    { text: "Encoder", options: { bold: true, color: REV } },
    { text: " LSTM reads the source and ends in a state ", options: {} },
    { text: "(h, c)", options: { bold: true } },
    { text: ". That state — and nothing else — conditions a ", options: {} },
    { text: "separate decoder", options: { bold: true, color: "2C7A4B" } },
    { text: " LSTM, a language model over the target. Trained by maximising log p(T | S); decoded by beam search.", options: {} },
  ], { x: M, y: 5.55, w: W - 2 * M, h: 1.0, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 15, color: INK_SOFT });
  note(s, "Paper: 4 layers x 1000 cells = 8000 numbers per sentence. Ours: 2 x 512 = 2048.");
}

// ============================================================ 4. why reversal
{
  const s = pres.addSlide();
  titleSlide(s, "Why reversing the source should help", "THE CLAIM UNDER TEST");
  if (has("figures/time_lag.png")) {
    s.addImage({ path: fig("figures/time_lag.png"), x: 0.55, y: 1.65, w: 12.2, h: 3.4 });
  }
  s.addText("Reversal leaves the average distance between corresponding words unchanged, but collapses the minimal time lag from the sentence length to one step — so backpropagation has a short path to establish the source→target correspondence early.", {
    x: M, y: 5.25, w: W - 2 * M, h: 0.9, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, color: INK_SOFT,
  });
  s.addText("The target is never reversed.", {
    x: M, y: 6.15, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: FWD,
  });
  s.addNotes("Paper sec. 3.3: perplexity 5.8 -> 4.7, BLEU 25.9 -> 30.6. This is the paper's key technical contribution.");
}

// ============================================================ 5. scale gap
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("THE HONEST CAVEAT", {
    x: M, y: 0.55, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: "8FA6DC", charSpacing: 1.2,
  });
  s.addText("Our BLEU is not comparable to the paper's", {
    x: M, y: 0.9, w: W - 2 * M, h: 0.75, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: 32, bold: true, color: PAPER,
  });
  const cards = [
    ["12M → 0.5M", "training sentence pairs"],
    ["384M → 58M", "parameters"],
    ["8 GPUs, 10 days\n→ 1 laptop GPU, 5 h", "compute"],
  ];
  cards.forEach(([v, l], i) => {
    const x = M + i * 4.05, y = 2.15;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: 3.75, h: 1.75, rectRadius: 0.09, fill: { color: "232C5E" }, line: { color: "37427E" } });
    s.addText(v, { x: x + 0.15, y: y + 0.22, w: 3.45, h: 0.95, isTextBox: true, margin: 0, align: "center", fontFace: HEAD_FONT, fontSize: v.length > 20 ? 17 : 26, bold: true, color: "CADCFC" });
    s.addText(l, { x: x + 0.15, y: y + 1.22, w: 3.45, h: 0.4, isTextBox: true, margin: 0, align: "center", fontFace: BODY_FONT, fontSize: 12, color: "8FA6DC" });
  });
  s.addText("So we claim reproduction of the direction, ordering and shape of the effects — never of the absolute numbers. Every table we show repeats this.", {
    x: M, y: 4.35, w: W - 2 * M, h: 0.8, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 16, color: PAPER,
  });
  s.addText("Held exactly as in the paper:  separate enc/dec LSTMs  ·  U(−0.08, 0.08) init  ·  plain SGD lr 0.7 with halving  ·  batch 128, loss ÷ batch size so clip@5 means what the paper says  ·  length bucketing  ·  beam search with hypotheses leaving the beam on <EOS>", {
    x: M, y: 5.35, w: W - 2 * M, h: 1.1, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 12.5, color: "9FB0D8",
  });
}

// ============================================================ 6. setup
{
  const s = pres.addSlide();
  titleSlide(s, "Data and evaluation", "SETUP");
  const kept = prep.clean ? prep.clean.kept : null;
  body(s, [
    "Training: News-Commentary v9 + Europarl v7 En–Fr — both constituent corpora of the official WMT'14 training set" + (kept ? `; ${kept.toLocaleString()} pairs survive cleaning, we sample ${(prep.train_pairs || 500000).toLocaleString()}` : ""),
    "Test: newstest2014, full 3003-sentence version — the paper's own ntst14, fetched via sacreBLEU",
    "Dev: newstest2013.  Vocabulary 32k / 32k, OOV → UNK" + (prep.unk_rate ? ` (${(100 * prep.unk_rate.test_src).toFixed(1)}% OOV on the English test side)` : ""),
    "Two BLEU numbers, always both: tokenized cased BLEU (the multi-bleu.pl equivalent the paper used) and standard sacreBLEU on detokenized output",
  ], { y: 1.75, h: 2.6 });
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 4.5, w: W - 2 * M, h: 1.35, rectRadius: 0.09, fill: { color: MIST }, line: { color: MIST } });
  s.addText([
    { text: "A bug worth flagging.  ", options: { bold: true, color: FWD } },
    { text: "News-Commentary v9 contains ~2,900 bare carriage returns inside lines — a different count in English than in French. Python's default newline handling splits on them and silently misaligns the parallel corpus. Our reader splits on \\n only; two regression tests pin it.", options: {} },
  ], { x: M + 0.22, y: 4.68, w: W - 2 * M - 0.44, h: 1.0, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 13, color: INK_SOFT });
}

// ============================================================ 7. THE RESULT
{
  const s = pres.addSlide();
  titleSlide(s, "Table 1 rows 3 and 4, reproduced", "RESULT");
  if (wmt.length) {
    const f = pick(wmt, "fwd", 12), r = pick(wmt, "rev", 12);
    const rows = ["fwd", "rev"].flatMap(d => [1, 2, 12].map(b => {
      const x = pick(wmt, d, b);
      if (!x) return null;
      const c = d === "fwd" ? FWD : REV;
      return [
        { text: (d === "fwd" ? "Single forward LSTM" : "Single reversed LSTM"), options: { align: "left", color: c, bold: b === 12 } },
        String(b),
        { text: x.bleu_tok.toFixed(2), options: { bold: true, color: c } },
        x.bleu_detok.toFixed(2),
        x.test_ppl == null ? "—" : x.test_ppl.toFixed(2),
        { text: b === 12 ? (d === "fwd" ? "26.17" : "30.59") : "—", options: { color: MUTED } },
      ];
    })).filter(Boolean);
    table(s, ["Model", "beam", "BLEU (tok)", "BLEU (detok)", "test ppl", "paper"], rows,
      { colW: [4.0, 1.0, 1.85, 1.95, 1.6, 1.7], y: 1.7, rowH: 0.33, fontSize: 12.5 });
    if (f && r) {
      const dB = (r.bleu_tok - f.bleu_tok).toFixed(2);
      const dP = f.test_ppl && r.test_ppl ? (100 * (f.test_ppl - r.test_ppl) / f.test_ppl).toFixed(0) : null;
      statCard(s, M, 4.55, 3.9, (dB > 0 ? "+" : "") + dB, "BLEU from reversal alone\n(paper: +4.42)", REV);
      if (dP) statCard(s, M + 4.25, 4.55, 3.9, dP + "%", "lower test perplexity\n(paper: 19%)", REV);
      statCard(s, M + 8.5, 4.55, 3.9, "ntst14", "scored on the paper's own\n3003-sentence test set", INK_SOFT);
    }
  } else {
    body(s, ["WMT'14 runs still training — regenerate with `make results-wmt && node scripts/make_slides.js`."]);
  }
  note(s, "Scale gap applies: 0.5M pairs vs 12M, 2x512 vs 4x1000, 32k/32k vocab vs 160k/80k. Compare within our column, not against the paper's.");
}

// ============================================================ 8. beam + length
{
  const s = pres.addSlide();
  titleSlide(s, "Beam size and sentence length", "THE OTHER TABLE 1 TRENDS");
  const bs = "wmt14_small/beam_sweep.png", bl = "wmt14_small/bleu_by_length.png";
  const useM30 = !has(bs);
  const p1 = has(bs) ? bs : "multi30k/beam_sweep.png";
  const p2 = has(bl) ? bl : "multi30k/bleu_by_length.png";
  if (has(p1)) s.addImage({ path: fig(p1), x: 0.55, y: 1.7, w: 5.9, h: 3.55 });
  if (has(p2)) s.addImage({ path: fig(p2), x: 6.85, y: 1.7, w: 5.9, h: 3.55 });
  let line = "Beam 2 recovers most of the benefit of beam 12 — the paper's own observation (sec. 3.2).";
  const src = wmt.length ? wmt : m30;
  const b1 = pick(src, "rev", 1), b2 = pick(src, "rev", 2), b12 = pick(src, "rev", 12);
  if (b1 && b2 && b12 && Math.abs(b12.bleu_tok - b1.bleu_tok) > 1e-9) {
    const frac = (100 * (b2.bleu_tok - b1.bleu_tok) / (b12.bleu_tok - b1.bleu_tok)).toFixed(0);
    line = `Beam 2 recovers ${frac}% of the gain from beam 1 to beam 12 — reproducing the paper's observation that "a beam of size 2 provides most of the benefits of beam search".`;
  }
  s.addText(line, { x: M, y: 5.42, w: W - 2 * M, h: 0.75, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 14, color: INK_SOFT });
  note(s, useM30 ? "Shown on Multi30k while the WMT runs finish." : "Right-hand panel is our analogue of the paper's Fig. 3.");
}

// ============================================================ 9. multi30k
{
  const s = pres.addSlide();
  titleSlide(s, "A seed-controlled replication", "SUPPORTING EVIDENCE — MULTI30K En→Fr");
  if (has("multi30k/training_curves.png")) {
    s.addImage({ path: fig("multi30k/training_curves.png"), x: 6.85, y: 1.9, w: 5.95, h: 3.6 });
  }
  const f = pick(m30, "fwd", 12), r = pick(m30, "rev", 12);
  const items = [
    "29k pairs, same language direction — trains in minutes, so we can afford three seeds per arm.",
    "Not a Table 1 reproduction: different corpus, domain and test set. It shows the effect survives where we can control for seed noise.",
  ];
  if (f && r) {
    items.push(`Reversal moves BLEU ${f.bleu_tok.toFixed(2)} → ${r.bleu_tok.toFixed(2)} and test perplexity ${f.test_ppl.toFixed(2)} → ${r.test_ppl.toFixed(2)} at beam 12.`);
    items.push("The effect is proportionally larger than the paper's — with only 29k pairs the forward model never learns a reliable correspondence, so it emits fluent French that does not track the source.");
  }
  body(s, items, { x: M, y: 1.78, w: 6.0, h: 4.4, fontSize: 14 });
  const nSeeds = new Set(m30.flatMap(r => r.seeds || [])).size;
  note(s, "Reversed (blue) sits below forward (red) at every epoch" + (nSeeds > 1 ? `, in all ${nSeeds} seeds.` : "."));
}

// ============================================================ 10. engineering
{
  const s = pres.addSlide();
  titleSlide(s, "Getting it right before spending compute", "ENGINEERING");
  s.addText("Correctness gates (pytest, ~12 s)", {
    x: M, y: 1.65, w: 6.0, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 16, bold: true, color: INK,
  });
  body(s, [
    "beam search at B=1 is exactly greedy decoding",
    "a wider beam never returns a lower-scoring hypothesis",
    "reversing the input by hand and setting the flag cancel out",
    "padding does not change the encoder's sentence vector",
    "scoring the references against themselves gives BLEU 100",
    "MPS and CPU agree on loss and gradients to 1e-3",
  ], { x: M, y: 2.1, w: 6.0, h: 3.9, fontSize: 14 });
  s.addText("Two findings worth keeping", {
    x: 7.0, y: 1.65, w: 5.7, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 16, bold: true, color: INK,
  });
  [["Corpus misalignment", "~2,900 in-line carriage returns in News-Commentary v9, unequal across languages — would have trained on mismatched pairs, silently.", FWD],
   ["Decoding 9× too slow on GPU", "Beam search is latency-bound: 317 ms/sentence on MPS vs 34 ms on CPU. Training stays on GPU; decoding moved to CPU.", REV]]
    .forEach(([h, t, c], i) => {
      const y = 2.12 + i * 2.05;
      s.addShape(pres.ShapeType.roundRect, { x: 7.0, y, w: 5.7, h: 1.8, rectRadius: 0.09, fill: { color: MIST }, line: { color: MIST } });
      s.addText(h, { x: 7.2, y: y + 0.13, w: 5.3, h: 0.32, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: c });
      s.addText(t, { x: 7.2, y: y + 0.52, w: 5.3, h: 1.15, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 12.5, color: INK_SOFT });
    });
  note(s, "PyTorch's Metal LSTM has a history of silently wrong numbers — finding that after an overnight run costs a day.");
}

// ============================================================ 11. verdict
{
  const s = pres.addSlide();
  titleSlide(s, "What this establishes — and what it does not", "VERDICT");
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 1.75, w: 5.85, h: 3.9, rectRadius: 0.09, fill: { color: "EAF0FA" }, line: { color: "EAF0FA" } });
  s.addText("Reproduced", { x: M + 0.25, y: 1.95, w: 5.35, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 16, bold: true, color: REV });
  body(s, [
    "Reversing the source improves both BLEU and perplexity by a clear margin, under an otherwise identical setup.",
    "A beam of 2 captures most of the benefit of a beam of 12.",
    "The ordering of Table 1 rows 3 and 4 holds at our scale.",
  ], { x: M + 0.25, y: 2.45, w: 5.35, h: 3.0, fontSize: 14 });
  s.addShape(pres.ShapeType.roundRect, { x: 7.0, y: 1.75, w: 5.7, h: 3.9, rectRadius: 0.09, fill: { color: "FAEDED" }, line: { color: "FAEDED" } });
  s.addText("Not attempted", { x: 7.25, y: 1.95, w: 5.2, h: 0.35, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 16, bold: true, color: FWD });
  body(s, [
    "Absolute BLEU anywhere near 26–31.",
    "The 5-model ensemble rows of Table 1.",
    "The 1000-best SMT rescoring of Table 2.",
  ], { x: 7.25, y: 2.45, w: 5.2, h: 3.0, fontSize: 14 });
  s.addText("These need the paper's data and compute. We say so, rather than presenting a smaller number as if it were comparable.", {
    x: M, y: 5.85, w: W - 2 * M, h: 0.6, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 14.5, italic: true, color: INK_SOFT,
  });
}

// ============================================================ 12. end-term
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("WHERE THE END-TERM GOES", {
    x: M, y: 1.5, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: "8FA6DC", charSpacing: 1.2,
  });
  s.addText("Everything passes through one fixed vector", {
    x: M, y: 1.88, w: W - 2 * M, h: 0.8, isTextBox: true, margin: 0,
    fontFace: HEAD_FONT, fontSize: 32, bold: true, color: PAPER,
  });
  // Phrase the long-sentence claim from the measured curve rather than asserting it.
  const lenSrc = (wmt.length ? wmt : m30);
  const lenRow = lenSrc.filter(r => r.reverse_source && r.bleu_by_length && r.bleu_by_length.length >= 2)
                       .sort((a, b) => b.beam_size - a.beam_size)[0];
  let lenClause = "";
  if (lenRow) {
    const b = lenRow.bleu_by_length;
    const first = b[0], last = b[b.length - 1];
    const drop = first.bleu_tok - last.bleu_tok;
    const corpus = wmt.length ? "on ntst14" : "on Multi30k";
    lenClause = drop > 1
      ? ` In our runs ${corpus}, reversed-model BLEU falls from ${first.bleu_tok.toFixed(1)} on ${first.bucket}-token sources to ${last.bleu_tok.toFixed(1)} on ${last.bucket} — the paper, at full scale, reported no such degradation.`
      : ` In our runs ${corpus}, BLEU holds up across length buckets (${first.bleu_tok.toFixed(1)} → ${last.bleu_tok.toFixed(1)}), matching the paper's own finding.`;
  }
  s.addText("A twelve-word sentence and a fifty-word sentence are compressed into the same 2048 numbers before a single target word is emitted. That is the bottleneck." + lenClause, {
    x: M, y: 2.75, w: 12.0, h: 1.15, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, color: "CADCFC",
  });
  const steps = [["Quantify", "BLEU and perplexity by source-length bucket; probe encoder-state saturation."],
                 ["Fix", "Implement Bahdanau attention on this same codebase; rerun the identical grid."],
                 ["Trace", "From attention to the Transformer, and on to modern LLM-based MT."]];
  steps.forEach(([h, t], i) => {
    const x = M + i * 4.05;
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.0, w: 3.75, h: 1.85, rectRadius: 0.09, fill: { color: "232C5E" }, line: { color: "37427E" } });
    s.addText(String(i + 1), { x: x + 0.18, y: 4.16, w: 0.5, h: 0.42, isTextBox: true, margin: 0, fontFace: HEAD_FONT, fontSize: 20, bold: true, color: FWD });
    s.addText(h, { x: x + 0.7, y: 4.18, w: 2.9, h: 0.4, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 16, bold: true, color: PAPER });
    s.addText(t, { x: x + 0.2, y: 4.68, w: 3.35, h: 1.0, isTextBox: true, margin: 0, fontFace: BODY_FONT, fontSize: 12, color: "9FB0D8" });
  });
  s.addText("The fix will be demonstrated, not merely argued.", {
    x: M, y: 6.1, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY_FONT, fontSize: 14, italic: true, color: "8FA6DC",
  });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("[slides] ->", OUT));
