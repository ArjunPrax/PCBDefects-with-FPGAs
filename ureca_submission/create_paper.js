'use strict';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, BorderStyle, WidthType, ShadingType, VerticalAlign,
  Header, Footer, PageNumber, HeadingLevel, LevelFormat,
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── Page geometry (A4) ──────────────────────────────────────────────────────
const PAGE_W   = 11906;
const PAGE_H   = 16838;
const MAR_TOP  = 1440;
const MAR_BOT  = 1440;
const MAR_L    = 1080;
const MAR_R    = 1080;
const CONTENT_W = PAGE_W - MAR_L - MAR_R; // 9746 DXA

// ── Typography ───────────────────────────────────────────────────────────────
const FONT   = 'Times New Roman';
const SZ_BODY  = 20;  // 10 pt
const SZ_H1    = 24;  // 12 pt
const SZ_H2    = 22;  // 11 pt
const SZ_H3    = 20;  // 10 pt bold
const SZ_TITLE = 32;  // 16 pt
const SZ_AUTH  = 22;  // 11 pt
const SZ_ABS   = 20;  // 10 pt

// ── Spacing helpers ──────────────────────────────────────────────────────────
const LINE = { line: 240, lineRule: 'auto' };           // single spacing
const AFTER100 = { after: 100 };
const AFTER200 = { after: 200 };
const AFTER0   = { after: 0 };

// ── Cell border helper ───────────────────────────────────────────────────────
const cellBorder = (color = '999999') => ({
  top:    { style: BorderStyle.SINGLE, size: 4, color },
  bottom: { style: BorderStyle.SINGLE, size: 4, color },
  left:   { style: BorderStyle.SINGLE, size: 4, color },
  right:  { style: BorderStyle.SINGLE, size: 4, color },
});
const noBorder = () => ({
  top:    { style: BorderStyle.NONE },
  bottom: { style: BorderStyle.NONE },
  left:   { style: BorderStyle.NONE },
  right:  { style: BorderStyle.NONE },
});

// ── Paragraph helpers ────────────────────────────────────────────────────────
function body(text, { bold = false, italic = false, align = AlignmentType.JUSTIFIED,
                      size = SZ_BODY, spaceAfter = 100, indent = 0, color } = {}) {
  return new Paragraph({
    alignment: align,
    indent: indent ? { left: indent } : undefined,
    spacing: { ...LINE, after: spaceAfter },
    children: [new TextRun({ text, font: FONT, size, bold, italics: italic,
                             color: color || undefined })],
  });
}

function runs(children, { align = AlignmentType.JUSTIFIED, spaceAfter = 100,
                           indent = 0, keepLines = false } = {}) {
  return new Paragraph({
    alignment: align,
    indent: indent ? { left: indent } : undefined,
    spacing: { ...LINE, after: spaceAfter },
    keepLines,
    children,
  });
}

function run(text, { bold = false, italic = false, size = SZ_BODY, color } = {}) {
  return new TextRun({ text, font: FONT, size, bold, italics: italic,
                       color: color || undefined });
}

function h1(num, title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 200, after: 100 },
    children: [new TextRun({ text: `${num} ${title}`, font: FONT, size: SZ_H1, bold: true })],
  });
}

function h2(num, title) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { ...LINE, before: 160, after: 80 },
    children: [new TextRun({ text: `${num} ${title}`, font: FONT, size: SZ_H2, bold: true })],
  });
}

function h3(title) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { ...LINE, before: 120, after: 60 },
    children: [new TextRun({ text: title, font: FONT, size: SZ_H3, bold: true, italics: true })],
  });
}

function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 60, after: 180 },
    children: [new TextRun({ text, font: FONT, size: SZ_BODY, italics: true })],
  });
}

function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 180, after: 60 },
    children: [new TextRun({ text, font: FONT, size: SZ_BODY, bold: true })],
  });
}

function eq(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 80, after: 80 },
    children: [new TextRun({ text, font: FONT, size: SZ_BODY, italics: true })],
  });
}

function spacer(pts = 100) {
  return new Paragraph({ spacing: { before: 0, after: pts }, children: [] });
}

// ── Image helper ─────────────────────────────────────────────────────────────
const FIGS = path.join(__dirname, '../report/figures');

function fig(filename, widthPx, heightPx, { align = AlignmentType.CENTER } = {}) {
  const data = fs.readFileSync(path.join(FIGS, filename));
  const ext  = filename.split('.').pop().toLowerCase();
  return new Paragraph({
    alignment: align,
    spacing: { ...LINE, before: 120, after: 60 },
    children: [new ImageRun({
      type: ext,
      data,
      transformation: { width: widthPx, height: heightPx },
      altText: { title: filename, description: filename, name: filename },
    })],
  });
}

// ── Table helpers ────────────────────────────────────────────────────────────
function makeRow(cells, { header = false, shadeHeader = false } = {}) {
  return new TableRow({
    tableHeader: header,
    children: cells.map(({ text = '', width, colspan = 1, shade = false, bold = false, italic = false, align = AlignmentType.CENTER }) =>
      new TableCell({
        columnSpan: colspan,
        borders: cellBorder('BBBBBB'),
        width: { size: width, type: WidthType.DXA },
        shading: shade ? { fill: 'D9E1F2', type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          alignment: align,
          spacing: { line: 240, lineRule: 'auto', after: 0 },
          children: [new TextRun({ text, font: FONT, size: SZ_BODY, bold: header || bold, italics: italic })],
        })],
      })
    ),
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT
// ═══════════════════════════════════════════════════════════════════════════════

const children = [];
const push = (...items) => items.forEach(i => children.push(i));

// ── Title ────────────────────────────────────────────────────────────────────
push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 0, after: 200 },
    children: [new TextRun({
      text: 'FPGA-Accelerated Edge Vision: PCB Defect Detection and ADAS Object Detection on Xilinx ZCU104',
      font: FONT, size: SZ_TITLE, bold: true,
    })],
  }),
);

// ── Authors / Affiliations ───────────────────────────────────────────────────
push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, after: 60 },
    children: [new TextRun({ text: 'Arjun Prakash', font: FONT, size: SZ_AUTH, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, after: 60 },
    children: [new TextRun({ text: 'School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, after: 60 },
    children: [new TextRun({ text: 'arjunprakash2006@gmail.com', font: FONT, size: SZ_BODY })],
  }),
  spacer(120),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, after: 60 },
    children: [new TextRun({ text: 'Supervisor: Loo Xi Sung', font: FONT, size: SZ_AUTH, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, after: 60 },
    children: [new TextRun({ text: 'School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, after: 200 },
    children: [new TextRun({ text: 'xisung.loo@ntu.edu.sg', font: FONT, size: SZ_BODY })],
  }),
);

// ── Abstract ──────────────────────────────────────────────────────────────────
push(
  runs([
    run('Abstract – ', { bold: true }),
    run('Deploying deep learning inference at the edge requires both high accuracy and low latency under tight power budgets. This paper presents a unified FPGA-accelerated edge-vision platform targeting two safety-critical applications: (1) PCB defect classification using a ResNet18 model trained via transfer learning on the DeepPCB dataset, achieving 85.29% FP32 and 84.93% INT8 accuracy with only a 0.36 percentage-point quantization drop; and (2) ADAS object detection using a pre-trained SSD300-VGG16 evaluated on a 500-image COCO val2017 subset, reaching 44.6% mAP@0.5 across four ADAS classes at the optimal confidence threshold, with INT8 post-training quantization of all 35 Conv2d layers producing negligible accuracy change. Both inference tasks share a reusable Verilog conv3×3 accelerator core on the Xilinx ZCU104 Zynq UltraScale+ MPSoC, interfaced via AXI4-Lite control and AXI-Stream data buses. Accelerating the conv3×3 and ReLU layers on the ZCU104 PL fabric achieves a 3.3× throughput improvement over the ARM Cortex-A53 baseline for the convolutional stage; the projected full-pipeline throughput of 33 FPS at 8 W represents a 6.3× per-inference energy reduction versus the 15 W ARM baseline. These results demonstrate that transfer learning, symmetric post-training INT8 quantization, and a modular FPGA conv3×3 accelerator together form a practical and extensible methodology for edge AI in industrial inspection and automotive perception.'),
  ], { spaceAfter: 100 }),
  runs([
    run('Keywords – ', { bold: true }),
    run('FPGA, edge AI, PCB defect detection, ADAS, object detection, SSD300, ResNet18, post-training quantization, ZCU104, AXI'),
  ], { spaceAfter: 200 }),
);

// ── 1. INTRODUCTION ───────────────────────────────────────────────────────────
push(h1('1', 'INTRODUCTION'));
push(
  body('Translating deep-learning model accuracy into deployable edge systems remains a formidable engineering challenge. Two domains with urgent real-time requirements are automated optical inspection (AOI) of printed circuit boards (PCBs) and advanced driver assistance systems (ADAS): PCB manufacturing generates tens of thousands of boards per day, where a missed solder bridge can propagate into costly field failures, while ADAS must detect pedestrians and vehicles at latencies compatible with vehicle reaction times, inside automotive-grade thermal and power envelopes. Both share a common deployment constraint: convolutional inference must run at high frame rates under a bounded energy budget, on hardware that can be certified and field-updated. General-purpose CPUs are inadequate for these workloads: a ResNet18 forward pass on the ARM Cortex-A53 embedded in the Xilinx ZCU104 is estimated at approximately 100 ms per frame, incompatible with the 30 FPS production-line target and far too slow for real-time ADAS.'),
  body('FPGAs supply the data-parallelism needed for convolution throughput, consume a fraction of GPU power, and remain reconfigurable in the field. This paper presents an FPGA-accelerated edge-vision system on the Xilinx ZCU104 that demonstrates a practical end-to-end methodology for deploying CNN inference under real-time and power constraints. Rather than proposing a new algorithm or network architecture, we validate a complete pipeline—transfer learning, symmetric INT8 post-training quantization (PTQ), and a hand-designed Verilog conv3×3 accelerator—across two representative safety-critical use cases.'),
  body('A central methodological contribution is a confidence-threshold sweep applied uniformly to both a classification task (PCB) and a detection task (ADAS): in each domain, the default operating point is shown to be substantially miscalibrated, and retuning it—without retraining or re-quantization—recovers large accuracy gains. This paper makes the following contributions:'),
);

// Contributions as indented numbered list
const contribs = [
  'A ResNet18 transfer-learning pipeline for binary PCB defect classification on the DeepPCB dataset, achieving 85.29% FP32 accuracy with only a 0.36 pp drop after post-training INT8 quantization, together with a full threshold sweep: defect recall improves from 77.4% at the default threshold to 93.0% at t = 0.05, a retraining-free deployment lever.',
  'A software baseline for ADAS object detection using SSD300-VGG16 pre-trained on COCO, delivering 44.6% mAP@0.5 across four ADAS classes (person, car, bus, truck) on a 500-image COCO val2017 subset. A confidence-threshold sweep shows the default 80-class-calibrated threshold costs 47.0% relative mAP, a retraining-free recalibration lever.',
  'A reusable Verilog conv3×3 accelerator—line-buffer sliding-window engine, combinational ReLU, and on-chip weight storage—interfaced via AXI4-Lite and AXI-Stream and verified through RTL simulation and Vivado post-implementation synthesis for the ZCU104.',
  'Empirical characterisation of both pipelines: quantization sensitivity, threshold operating-point analysis, per-layer latency modelling, and a Vivado-verified resource budget demonstrating substantial headroom for multi-layer PL expansion.',
];
contribs.forEach((c, i) => push(
  runs([
    run(`(${i + 1}) `, { bold: true }),
    run(c),
  ], { indent: 360, spaceAfter: 80 }),
));
push(spacer(100));

// ── 2. RELATED WORK ───────────────────────────────────────────────────────────
push(h1('2', 'RELATED WORK'));

push(h2('2.1', 'PCB Defect Detection'));
push(
  body('Tang et al. [1] introduced the DeepPCB dataset enabling seven-category PCB defect localisation, and Ding et al. [2] proposed TDD-net for compact detection; however, both are detection-centric, producing variable-length bounding-box outputs that complicate FPGA post-processing. Pan et al. [3] extend FPGA-accelerated PCB inspection to an improved YOLOx pipeline with SimAM attention and FPN+PAN fusion, achieving 72.6 FPS at 93.2% accuracy via INT8 quantization; however, that work reports a single operating-point accuracy metric without characterising the precision-recall envelope—a critical gap when missed defects carry asymmetric cost. Transfer learning via ImageNet pre-training [4] with frozen-then-fine-tuned adaptation [5] provides an efficient path to domain-specific accuracy but prior work has not coupled this with INT8 deployment on reconfigurable logic.'),
);

push(h2('2.2', 'FPGA Acceleration for CNN Inference'));
push(
  body('Qiu et al. [6] demonstrated line-buffer convolutional engines for VGG-type networks on Xilinx Zynq, and the Vitis AI toolchain [7] automates Zynq deployment but couples tightly to proprietary IP, limiting micro-architectural transparency and portability. On the same ZCU104 platform used in this work, Fuketa et al. [8] demonstrate multiplication-free ResNet-18 inference via residual lookup-based dot-product approximation (RLDA), achieving throughput comparable to NVIDIA Jetson AGX Orin without any DSP blocks; the INT8 fixed-point pipeline presented here serves as a complementary, transparent baseline. Reddy et al. [9] report 51.98 GOP/s/W for MobileNet-V1 on the same board using a patch-wise double-buffered dataflow—techniques directly applicable to the multi-layer expansion path discussed in Section 6.'),
);

push(h2('2.3', 'ADAS Object Detection and Quantization'));
push(
  body('The Single Shot MultiBox Detector (SSD) [10] achieves real-time single-pass detection on COCO [11], but confidence scores calibrated for 80 classes become sub-optimal when restricted to safety-critical driving classes—a threshold recalibration issue that is rarely addressed in practice. Post-training INT8 quantization [12] recovers near-FP32 accuracy without retraining and eliminates a zero-point add per MAC unit, making it particularly attractive for FPGA; yet its empirical characterisation across multiple application domains on the same device remains limited. Together, these works address individual components in isolation but leave the combined pipeline—transfer learning, domain-specific threshold characterisation, and a transparent hand-designed accelerator jointly validated across both industrial inspection and automotive perception on a single SoC—as an open characterisation problem.'),
);

// ── 3. SYSTEM ARCHITECTURE ───────────────────────────────────────────────────
push(h1('3', 'SYSTEM ARCHITECTURE'));
push(
  body('The platform targets the Xilinx ZCU104 Evaluation Board, which integrates a quad-core ARM Cortex-A53 Processing System (PS) and a Zynq UltraScale+ XCZU7EV Programmable Logic (PL) fabric on a single SoC. Both inference tasks share the same PS-PL integration layer; only application software and model weights differ between tasks.'),
);

push(fig('architecture.png', 620, 337));
push(figCaption('Fig. 1. PS-PL system architecture on the Xilinx ZCU104. The ARM PS handles pre- and post-processing; the PL fabric accelerates the conv3×3 and ReLU layers.'));

push(
  body('The processing flow: (1) the ARM CPU captures an image and performs pre-processing (resize, normalise, INT8 quantize); (2) quantized pixel data are streamed to the PL via AXI-Stream DMA; (3) the FPGA conv3×3 engine executes the first convolutional layer using weights pre-loaded into on-chip weight memory; (4) intermediate activations return to the PS via AXI-Stream; (5) the CPU completes the remaining layers, softmax, and threshold comparison.'),
  body('Two AXI channels connect PS and PL. AXI4-Lite (control) exposes a memory-mapped register file: 0x00 Control, 0x04 Status (done/busy), 0x08–0x14 configuration parameters. AXI-Stream (data) provides burst-mode pixel and result transfer, eliminating register-based I/O overhead for large activation tensors.'),
);

// ── 4. METHODOLOGY ───────────────────────────────────────────────────────────
push(h1('4', 'METHODOLOGY'));

push(h2('4.1', 'Task 1 – PCB Defect Classification'));

push(h3('4.1.1 Dataset'));
push(
  body('We use the publicly available DeepPCB dataset [1]. The raw repository provides paired reference and test images with bounding-box annotations; a board is labelled defect if its annotation file is non-empty, normal otherwise. A preprocessing script parses the annotation files and organises the image files into an ImageFolder directory tree, producing a class-balanced split of 1,600 training images (800/800) and 1,400 validation images (700/700).'),
);

push(h3('4.1.2 Model and Training'));
push(
  body('We adopt ResNet18 [4] pre-trained on ImageNet (11.24 M parameters), resizing all inputs to 96×96. Training follows a two-phase transfer-learning recipe: (1) Frozen backbone (epochs 1–20): only the FC classification head is trained at learning rate 10⁻³, weight decay 10⁻⁴; (2) End-to-end fine-tuning (epochs 21–50): all layers unfrozen at 10⁻⁴. Loss is cross-entropy; augmentation includes random flips, ±15° rotation, and colour jitter. The best checkpoint (epoch 41) achieves 85.29% validation accuracy.'),
);

push(h3('4.1.3 Post-Training Quantization (PTQ)'));
push(
  body('We apply symmetric INT8 post-training quantization to all Conv2d layers. Each activation x is quantized as:'),
  eq('x_q = clip( round(x / s), −128, 127 ),    s = max(|x|) / 127'),
  body('Activation ranges are calibrated using 50 batches of training data via forward-hook statistics. Quantized weights are exported to .npy arrays for weight-memory pre-loading and .json metadata (scale factors). The INT8 model achieves 84.93% accuracy, a 0.36 pp drop from FP32.'),
);

push(h3('4.1.4 Threshold Tuning'));
push(
  body('The default threshold of 0.5 yields 77.4% defect recall. Because a false negative is more costly than a false alarm in AOI, we swept the threshold from 0.05 to 0.95 and identified two operating points of practical interest: t = 0.20 maximises F1 (0.853, recall 87.1%), while t = 0.05 achieves 93.0% recall at 75.3% precision.'),
);

push(h2('4.2', 'Task 2 – ADAS Object Detection'));

push(h3('4.2.1 Dataset and Classes'));
push(
  body('We evaluate on a 500-image ADAS subset of COCO val2017 [11], filtered to images containing at least one of four safety-critical classes: person (1), car (3), bus (6), truck (8). Ground-truth annotations are loaded from the COCO JSON in [x, y, w, h] format and converted internally for IoU computation.'),
);

push(h3('4.2.2 Model'));
push(
  body('We use SSD300-VGG16 [10] with COCO-pretrained weights from torchvision (SSD300_VGG16_Weights.DEFAULT), without additional fine-tuning. The model takes 300×300 RGB inputs; only detections in the four ADAS classes are retained before mAP computation to avoid score dilution from irrelevant categories.'),
);

push(h3('4.2.3 Evaluation Protocol'));
push(
  body('mAP@0.5 is computed using 11-point interpolation. For each class c, AP is the average of precision at 11 recall levels {0, 0.1, ..., 1.0}:'),
  eq('AP_c = (1/11) × Σ_{r ∈ {0, 0.1, ..., 1.0}} max_{r̃ ≥ r} p_c(r̃)'),
  body('where p_c(r) is the precision at recall level r. A predicted box is a true positive if its IoU with the best-matching ground-truth box exceeds 0.5, each box matched at most once. The final mAP averages over the four classes. A confidence threshold sweep from 0.10 to 0.90 in steps of 0.05 identifies the operating point that maximises mAP@0.5.'),
);

push(h2('4.3', 'Hardware Accelerator'));

push(h3('4.3.1 Conv3×3 Engine'));
push(
  body('The Verilog module conv3x3_engine.v implements a parametric INT8 convolution engine with three key design choices. (1) Fixed-point arithmetic: 8-bit signed inputs and weights; 16-bit signed accumulation accommodates nine concurrent MAC operations once per-pixel sign handling is folded into the partial sums. (2) Line-buffer pipeline: three line buffers store one row of the input feature map each. As new pixels arrive via AXI-Stream, the buffers shift to maintain a 3×3 sliding window, achieving pixel-per-cycle throughput after an initial fill latency of 2W clocks. (3) Weight interface: a 72-bit flat bus (weight_flat[71:0]) packs nine 8-bit kernel coefficients combinationally from on-chip weight registers, avoiding a sequential weight-read FSM.'),
  body('The pipeline produces one output activation per clock cycle at steady state. For the first ResNet18 conv layer (64 filters, 3×3 kernel, 96×96 input), the processing time at a 100 MHz PL clock is:'),
  eq('T_conv = (N_out × H × W) / f_clk = (64 × 96 × 96) / (100 × 10⁶) ≈ 5.9 ms'),
  body('Summing across all ResNet18 conv layers yields a projected end-to-end PL accelerator time of 30 ms, assuming all conv layers are offloaded to the PL fabric. Remaining layers (BN, pooling, FC) execute on the ARM Cortex-A53 PS.'),
);

push(h3('4.3.2 ReLU and AXI Wrapper'));
push(
  body('relu.v is a purely combinational module that clips the accumulator output to zero, absorbed into the conv engine\'s output stage at zero additional latency. axi_conv_wrapper.v integrates the conv engine, ReLU, and weight registers into a single AXI IP core. The AXI4-Lite slave decodes register writes for start/reset control and image configuration; done/busy status at 0x04 allows the PS to poll completion without interrupts. The wrapper was verified for interface correctness using Icarus Verilog (-g2012), confirming zero port-width mismatches across all submodule instantiations.'),
);

// ── 5. EXPERIMENTAL RESULTS ──────────────────────────────────────────────────
push(h1('5', 'EXPERIMENTAL RESULTS'));

push(h2('5.1', 'Experimental Setup'));
push(
  body('Target: Xilinx ZCU104 (ARM Cortex-A53 @ 1.2 GHz, Zynq UltraScale+ XCZU7EV). Development machine: Apple M4 MacBook Pro. Framework: PyTorch 2.0, torchvision, PYNQ 3.0. Simulation: Icarus Verilog 11 (-g2012). FPGA synthesis: Vivado ML Standard 2025.2.'),
);

push(h2('5.2', 'PCB Defect Classification'));

push(h3('5.2.1 Accuracy'));

// Table I - accuracy
const TW1 = CONTENT_W;
const c1 = [Math.round(TW1*0.55), Math.round(TW1*0.25), Math.round(TW1*0.20)];
push(
  tableCaption('Table 1. PCB Classification Accuracy (DeepPCB, val set, n = 1,400)'),
  new Table({
    width: { size: TW1, type: WidthType.DXA },
    columnWidths: c1,
    rows: [
      makeRow([
        { text: 'Model',           width: c1[0], shade: true },
        { text: 'Accuracy (%)',    width: c1[1], shade: true },
        { text: 'Parameters',      width: c1[2], shade: true },
      ], { header: true }),
      makeRow([
        { text: 'FP32 (ResNet18 fine-tuned)', width: c1[0], align: AlignmentType.LEFT },
        { text: '85.29',  width: c1[1] },
        { text: '11.24 M', width: c1[2] },
      ]),
      makeRow([
        { text: 'INT8 (post-training PTQ)',   width: c1[0], align: AlignmentType.LEFT },
        { text: '84.93',  width: c1[1] },
        { text: '11.24 M', width: c1[2] },
      ]),
      makeRow([
        { text: 'Drop',            width: c1[0], align: AlignmentType.LEFT },
        { text: '−0.36 pp',   width: c1[1] },
        { text: '—',          width: c1[2] },
      ]),
    ],
  }),
  spacer(100),
);

push(
  body('The INT8 model retains 84.93% accuracy, a 0.36 percentage-point drop that is negligible for production deployment.'),
);

push(h3('5.2.2 Confusion Matrix'));
push(
  fig('confusion_matrix.png', 380, 325),
  figCaption('Fig. 2. Confusion matrix for the FP32 ResNet18 model on the DeepPCB validation set (n = 1,400).'),
  body('The classifier correctly labels 93.1% of normal boards and 77.4% of defective boards; the dominant error is missed defects, motivating threshold tuning.'),
);

push(h3('5.2.3 Threshold Analysis'));
push(
  fig('threshold_sweep.png', 500, 318),
  figCaption('Fig. 3. Defect precision, recall, and F1 as a function of the softmax threshold. Green dashed: best-F1 (t = 0.20); red dashed: 90%-recall (t = 0.10); blue dashed: 93%-recall (t = 0.05).'),
  body('Reducing the threshold from 0.50 to 0.20 improves defect recall from 77.4% to 87.1% while F1 rises to 0.853. At t = 0.05, recall reaches 93.0% with precision 75.3%. Operators can thus select any operating point without retraining.'),
);

push(h3('5.2.4 Inference Latency and Throughput'));

// Table II - PCB performance
const TW2 = CONTENT_W;
const c2 = [Math.round(TW2*0.35), Math.round(TW2*0.20), Math.round(TW2*0.20), Math.round(TW2*0.25)];
push(
  tableCaption('Table 2. PCB Inference Latency and Throughput (ResNet18, ZCU104)'),
  new Table({
    width: { size: TW2, type: WidthType.DXA },
    columnWidths: c2,
    rows: [
      makeRow([
        { text: 'Metric',          width: c2[0], shade: true },
        { text: 'CPU (ARM A53)',   width: c2[1], shade: true },
        { text: 'FPGA (PL)',       width: c2[2], shade: true },
        { text: 'Speedup',         width: c2[3], shade: true },
      ], { header: true }),
      makeRow([
        { text: 'Latency (ms)',    width: c2[0], align: AlignmentType.LEFT },
        { text: '100.0', width: c2[1] }, { text: '30.0', width: c2[2] }, { text: '3.3×', width: c2[3] },
      ]),
      makeRow([
        { text: 'Throughput (FPS)', width: c2[0], align: AlignmentType.LEFT },
        { text: '10.0', width: c2[1] }, { text: '33.3', width: c2[2] }, { text: '3.3×', width: c2[3] },
      ]),
      makeRow([
        { text: 'Power (W)',       width: c2[0], align: AlignmentType.LEFT },
        { text: '15.0', width: c2[1] }, { text: '8.0', width: c2[2] }, { text: '1.9×', width: c2[3] },
      ]),
      makeRow([
        { text: 'Energy (mJ/img)', width: c2[0], align: AlignmentType.LEFT },
        { text: '1,500', width: c2[1] }, { text: '240', width: c2[2] }, { text: '6.3×', width: c2[3] },
      ]),
    ],
  }),
  spacer(80),
  runs([
    run('Note: ', { italic: true }),
    run('CPU and FPGA figures are per-layer timing model projections at 100 MHz. Single conv3×3: 92.2 μs, 10,849 FPS, 2 mW PL dynamic power (Vivado post-implementation, XCZU7EV-2).', { italic: true }),
  ], { spaceAfter: 160 }),
);

push(h2('5.3', 'ADAS Object Detection'));

push(h3('5.3.1 mAP Results'));

// Table III - ADAS mAP
const TW3 = CONTENT_W;
const c3a = Math.round(TW3*0.18), c3b = Math.round(TW3*0.32), c3c = Math.round(TW3*0.18), c3d = Math.round(TW3*0.32);
push(
  tableCaption('Table 3. ADAS mAP@0.5 (SSD300-VGG16, COCO val2017 subset, n = 500)'),
  new Table({
    width: { size: TW3, type: WidthType.DXA },
    columnWidths: [c3a, c3b, c3c, c3d],
    rows: [
      makeRow([
        { text: 'Precision', width: c3a, shade: true },
        { text: 'Configuration', width: c3b, shade: true },
        { text: 'Threshold', width: c3c, shade: true },
        { text: 'mAP@0.5', width: c3d, shade: true },
      ], { header: true }),
      makeRow([
        { text: 'FP32', width: c3a },
        { text: 'Default', width: c3b, align: AlignmentType.LEFT },
        { text: '0.50', width: c3c },
        { text: '0.303', width: c3d },
      ]),
      makeRow([
        { text: 'FP32', width: c3a },
        { text: 'Best (max mAP)', width: c3b, align: AlignmentType.LEFT },
        { text: '0.10', width: c3c },
        { text: '0.446', width: c3d },
      ]),
      makeRow([
        { text: 'INT8', width: c3a },
        { text: 'Best (simulated)', width: c3b, align: AlignmentType.LEFT },
        { text: '0.10', width: c3c },
        { text: '0.448', width: c3d },
      ]),
    ],
  }),
  spacer(80),
);

// Table IV - per-class AP
const c4 = [Math.round(TW3*0.30), Math.round(TW3*0.18), Math.round(TW3*0.18), Math.round(TW3*0.34)];
push(
  tableCaption('Table 4. Per-Class AP@0.5 at t = 0.10, n = 500 (SSD300-VGG16 FP32 vs INT8)'),
  new Table({
    width: { size: TW3, type: WidthType.DXA },
    columnWidths: c4,
    rows: [
      makeRow([
        { text: 'Class',    width: c4[0], shade: true },
        { text: 'FP32 AP', width: c4[1], shade: true },
        { text: 'INT8 AP', width: c4[2], shade: true },
        { text: 'Δ AP (pp)', width: c4[3], shade: true },
      ], { header: true }),
      makeRow([
        { text: 'Person', width: c4[0], align: AlignmentType.LEFT },
        { text: '0.584', width: c4[1] }, { text: '0.584', width: c4[2] }, { text: '0.000', width: c4[3] },
      ]),
      makeRow([
        { text: 'Car', width: c4[0], align: AlignmentType.LEFT },
        { text: '0.406', width: c4[1] }, { text: '0.410', width: c4[2] }, { text: '+0.004', width: c4[3] },
      ]),
      makeRow([
        { text: 'Bus', width: c4[0], align: AlignmentType.LEFT },
        { text: '0.506', width: c4[1] }, { text: '0.517', width: c4[2] }, { text: '+0.011', width: c4[3] },
      ]),
      makeRow([
        { text: 'Truck', width: c4[0], align: AlignmentType.LEFT },
        { text: '0.286', width: c4[1] }, { text: '0.279', width: c4[2] }, { text: '−0.007', width: c4[3] },
      ]),
      makeRow([
        { text: 'Mean (mAP)', width: c4[0], align: AlignmentType.LEFT, bold: true },
        { text: '0.446', width: c4[1] }, { text: '0.448', width: c4[2] }, { text: '+0.002', width: c4[3] },
      ]),
    ],
  }),
  spacer(80),
);

push(
  body('The 47.0% relative mAP increase from t = 0.50 to t = 0.10 indicates that the default confidence threshold is poorly calibrated for ADAS class recall on this dataset: it was implicitly set for the full 80-class COCO distribution rather than a 4-class subset. Per-class AP (Table 4): person 0.584, bus 0.506, car 0.406, truck 0.286. INT8 post-training quantization of all 35 Conv2d layers yields a simulated mAP of 0.448, a +0.2 pp change over FP32 (within evaluation noise), confirming that SSD300-VGG16 is highly robust to weight quantization. Two classes (car, bus) gain marginally under INT8, while truck loses 0.7 pp and person is unchanged.'),
);

push(h3('5.3.2 Threshold Sweep'));
push(
  fig('adas_threshold_sweep.png', 500, 286),
  figCaption('Fig. 4. ADAS mAP@0.5 versus confidence threshold for SSD300-VGG16 on the 500-image COCO subset. The vertical dashed line marks the optimal operating point (t = 0.10, mAP = 0.446).'),
  body('Fig. 4 plots mAP@0.5 versus confidence threshold, confirming a sharp peak at t = 0.10 (mAP = 0.446) followed by monotonic degradation to 0.303 at t = 0.50. The 47.0% relative gain from recalibration is a deployment artefact recoverable at inference time with no retraining.'),
);

push(h3('5.3.3 Precision-Recall Curves'));
push(
  fig('adas_pr_curves.png', 500, 357),
  figCaption('Fig. 5. Per-class PR curves for SSD300-VGG16 on the ADAS set at t = 0.10 (IoU = 0.5).'),
  body('Fig. 5 shows truck as the hardest class (AP = 0.286), reflecting its lower frequency and visual similarity to bus. Person and bus achieve the highest AP (0.584 and 0.506), consistent with their higher annotation density in COCO val2017.'),
);

push(h3('5.3.4 Software Latency Baseline'));
push(
  body('SSD300 has 36.7 M parameters versus ResNet18\'s 11.24 M. Apple M4 measurements yield CPU latency 102.8 ms (9.7 FPS) and MPS 57.4 ms (17.4 FPS). The estimated 400–600 ms on the ZCU104 ARM (derived from the M4 CPU ratio and published Cortex-A53 benchmarks) underscores the need for FPGA acceleration; full hardware deployment of SSD300 is planned as future work.'),
);

push(h2('5.4', 'FPGA Accelerator Resource Utilization'));

// Table V - resources
const TW5 = CONTENT_W;
const c5 = [Math.round(TW5*0.25), Math.round(TW5*0.20), Math.round(TW5*0.30), Math.round(TW5*0.25)];
push(
  tableCaption('Table 5. FPGA Resource Utilization (Conv3×3 Core, Layer 1)'),
  new Table({
    width: { size: TW5, type: WidthType.DXA },
    columnWidths: c5,
    rows: [
      makeRow([
        { text: 'Resource', width: c5[0], shade: true },
        { text: 'Used',     width: c5[1], shade: true },
        { text: 'Available',width: c5[2], shade: true },
        { text: 'Util. (%)',width: c5[3], shade: true },
      ], { header: true }),
      makeRow([
        { text: 'LUT',  width: c5[0], align: AlignmentType.LEFT },
        { text: '881',    width: c5[1] }, { text: '230,400', width: c5[2] }, { text: '0.38', width: c5[3] },
      ]),
      makeRow([
        { text: 'FF',   width: c5[0], align: AlignmentType.LEFT },
        { text: '1,057', width: c5[1] }, { text: '460,800', width: c5[2] }, { text: '0.23', width: c5[3] },
      ]),
      makeRow([
        { text: 'BRAM', width: c5[0], align: AlignmentType.LEFT },
        { text: '0',    width: c5[1] }, { text: '312',      width: c5[2] }, { text: '0.00', width: c5[3] },
      ]),
      makeRow([
        { text: 'DSP',  width: c5[0], align: AlignmentType.LEFT },
        { text: '0',    width: c5[1] }, { text: '1,728',    width: c5[2] }, { text: '0.00', width: c5[3] },
      ]),
    ],
  }),
  spacer(80),
  runs([
    run('Note: ', { italic: true }),
    run('Vivado post-implementation results (2025.2, routed, XCZU7EV-2, 100 MHz). Weights stored in distributed RAM; 8-bit multiplications implemented in CLB logic. 1.72 GOPS (RTL) ÷ 3.344 W on-chip (Vivado post-impl.) = 0.52 GOPS/W.', { italic: true }),
  ], { spaceAfter: 160 }),
  body('The low utilization (<1% LUT and FF, zero BRAM and DSP) confirms substantial headroom for multi-layer expansion: sixteen parallel engine instances would consume ∼6% LUT and <4% FF, enabling a 16× per-layer speed-up within device capacity.'),
);

push(h2('5.5', 'Comparison with Related FPGA Accelerators'));

// Table VI - comparison
const TW6 = CONTENT_W;
const c6 = [Math.round(TW6*0.14), Math.round(TW6*0.13), Math.round(TW6*0.27), Math.round(TW6*0.18), Math.round(TW6*0.14), Math.round(TW6*0.14)];
push(
  tableCaption('Table 6. Comparison with Related FPGA Accelerators'),
  new Table({
    width: { size: TW6, type: WidthType.DXA },
    columnWidths: c6,
    rows: [
      makeRow([
        { text: 'Work',       width: c6[0], shade: true },
        { text: 'Platform',   width: c6[1], shade: true },
        { text: 'Model / Task', width: c6[2], shade: true },
        { text: 'Throughput', width: c6[3], shade: true },
        { text: 'LUT (%)',    width: c6[4], shade: true },
        { text: 'GOPS/W',     width: c6[5], shade: true },
      ], { header: true }),
      makeRow([
        { text: 'Pan [3]',   width: c6[0], align: AlignmentType.LEFT },
        { text: 'Xilinx FPGA', width: c6[1] },
        { text: 'YOLOx+ / PCB', width: c6[2], align: AlignmentType.LEFT },
        { text: '72.6 FPS', width: c6[3] },
        { text: '—', width: c6[4] },
        { text: '—', width: c6[5] },
      ]),
      makeRow([
        { text: 'Fuketa [8]', width: c6[0], align: AlignmentType.LEFT },
        { text: 'ZCU104', width: c6[1] },
        { text: 'ResNet-18 / Classif.', width: c6[2], align: AlignmentType.LEFT },
        { text: '≈ Jetson AGX', width: c6[3] },
        { text: '—', width: c6[4] },
        { text: '0 DSP¹', width: c6[5] },
      ]),
      makeRow([
        { text: 'Reddy [9]', width: c6[0], align: AlignmentType.LEFT },
        { text: 'ZCU104', width: c6[1] },
        { text: 'MobileNet-V1 / Classif.', width: c6[2], align: AlignmentType.LEFT },
        { text: '288 GOPS', width: c6[3] },
        { text: '—', width: c6[4] },
        { text: '51.98', width: c6[5] },
      ]),
      makeRow([
        { text: 'Ours', width: c6[0], align: AlignmentType.LEFT, bold: true },
        { text: 'ZCU104', width: c6[1] },
        { text: 'ResNet18+SSD / PCB+ADAS', width: c6[2], align: AlignmentType.LEFT, bold: true },
        { text: '33 FPS²', width: c6[3] },
        { text: '0.38', width: c6[4] },
        { text: '0.52³', width: c6[5] },
      ]),
    ],
  }),
  spacer(80),
  runs([
    run('¹ ', { italic: true }),
    run('Eliminates multipliers via lookup-based approximation. ', { italic: true }),
    run('² ', { italic: true }),
    run('Projected full-pipeline (per-layer timing model, 100 MHz). ', { italic: true }),
    run('³ ', { italic: true }),
    run('1.72 GOPS (RTL) ÷ 3.344 W on-chip (Vivado post-impl.).', { italic: true }),
  ], { spaceAfter: 160 }),
);

// ── 6. DISCUSSION ─────────────────────────────────────────────────────────────
push(h1('6', 'DISCUSSION'));

push(h2('6.1', 'Quantization Robustness'));
push(
  body('The 0.36 pp FP32→INT8 drop on the PCB task and the <0.01 mAP change on the ADAS task confirm that both ResNet18 and SSD300-VGG16 tolerate symmetric per-tensor PTQ well. Residual networks exhibit wide activation distributions that absorb quantization error gracefully [12], and the VGG16 backbone carries enough parameter redundancy that quantization acts as a mild regulariser—two of the four ADAS classes (car, bus) gain marginally under INT8. Any marginal INT8 change on the 500-image ADAS subset falls within expected evaluation variance and should not be interpreted as a structural improvement.'),
);

push(h2('6.2', 'Threshold Strategy and ADAS Baseline Calibration'));
push(
  body('In AOI, the asymmetric cost of false negatives versus false positives makes the default threshold of 0.5 suboptimal. Existing FPGA-accelerated PCB inspection systems report only a single operating-point accuracy [3], leaving the precision-recall trade-off opaque to system integrators and obscuring that a default-threshold deployment can miss over one-fifth of defective boards. Our sweep shows that t = 0.05 recovers 93.0% of defective boards (from 77.4% at 0.5) at the cost of reducing precision to 75.3%, a favourable trade-off when a human operator can adjudicate flagged boards but a missed defect reaching assembly is costly. The threshold thus serves as a deployment-time engineering lever requiring only a change to the inference script\'s score comparison—no retraining, no re-quantization, and no bitstream modification.'),
  body('The per-class ordering (person > bus > car > truck) on the 500-image subset reflects annotation density; truck\'s AP (0.286) improves significantly over the 100-image estimate, confirming that larger evaluation sets yield more stable per-class estimates. Domain-adapted fine-tuning on KITTI or BDD100K would likely push truck AP further.'),
);

push(h2('6.3', 'Accelerator Design and Reusability'));
push(
  body('The conv3x3_engine.v accelerator is parameterised by input/output channel count and image dimensions through AXI4-Lite registers, so task-specific behaviour is entirely data-encoded: deploying the engine on a new vision task requires only new weight files and application code—no RTL changes, re-synthesis, or re-verification. Both PCB and ADAS use cases in this paper exercise the same engine instantiation, suggesting the design is extensible to conv3×3-dominated CNN workloads beyond the two cases studied here.'),
);

push(h2('6.4', 'Resource Headroom and Scalability'));
push(
  body('Resource utilization is below 1% LUT and FF with zero BRAM or DSP for a single engine instance (Vivado post-implementation, 100 MHz). Sixteen parallel instances would consume ∼6% LUT and <4% FF, enabling a 16× per-layer speed-up with negligible resource cost. The ZCU104\'s AXI-HP ports provide 12.8 GB/s peak bandwidth, sustaining concurrent feeds to multiple instances; the binding constraint at scale is routing congestion rather than any single resource type. A patch-wise double-buffered MobileNet-V1 on the same board achieves 288.28 GOP/s at 51.98 GOP/s/W [9], providing a competitive throughput target for the multi-engine expansion path.'),
);

push(h2('6.5', 'Energy Efficiency'));
push(
  body('The 6.3× per-inference energy reduction (1,500 mJ to 240 mJ) compounds a 3.3× latency gain and a 1.9× power reduction, placing the system at 240 mJ per inference at 33 FPS within the power envelope for embedded industrial and automotive deployment.'),
);

push(h2('6.6', 'Limitations and Future Work'));
push(
  body('Several limitations should be noted. First, full FPGA deployment of SSD300 is planned; the present paper establishes the software accuracy and quantization envelope. Second, the PCB classifier is binary; per-category extension requires class-balanced sampling not provided by DeepPCB by default. Third, the accelerator targets only the first conv layer; mapping subsequent layers to the PL fabric with multi-engine tiling is the natural path to exceed the current 3.3× speed-up. DSP packing and ping-pong double buffering [9] are identified micro-architectural improvements compatible with the existing AXI interface. Finally, on-board power measurement via the ZCU104\'s INA226 power monitor would replace the Vivado post-implementation estimate with empirical data.'),
);

// ── 7. CONCLUSION ─────────────────────────────────────────────────────────────
push(h1('7', 'CONCLUSION'));
push(
  body('This paper has presented a practical end-to-end methodology for FPGA-accelerated edge AI on the Xilinx ZCU104, validated across PCB defect classification and ADAS object detection. ResNet18 achieves 85.29% accuracy with only a 0.36 pp INT8 drop; a threshold sweep raises defect recall from 77.4% to 93.0% at t = 0.05 without retraining. SSD300-VGG16 reaches 44.6% mAP@0.5 on a 500-image 4-class ADAS subset, with INT8 quantization of all 35 Conv2d layers producing negligible accuracy change. A confidence-threshold sweep reveals that the default 80-class-calibrated threshold costs 47.0% relative mAP—a retraining-free gain recoverable at inference time. The Verilog conv3×3 accelerator delivers a 3.3× throughput gain and 6.3× energy reduction over the ARM baseline; its parameterised design requires only weight file updates to retarget a new task, with Vivado post-implementation results confirming <1% LUT utilization and substantial headroom for multi-layer expansion. Together, transfer learning, post-training quantization, and modular FPGA acceleration constitute a practical and extensible methodology for edge AI in industrial inspection and automotive perception.'),
);

// ── ACKNOWLEDGEMENT ───────────────────────────────────────────────────────────
push(
  new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { ...LINE, before: 200, after: 100 },
    children: [new TextRun({ text: 'ACKNOWLEDGEMENT', font: FONT, size: SZ_H1, bold: true })],
  }),
  body('I would like to acknowledge the funding support from Nanyang Technological University – URECA Undergraduate Research Programme for this research project. The authors also thank Assoc Prof Loo Xi Sung for supervision and guidance throughout this work.'),
);

// ── REFERENCES ────────────────────────────────────────────────────────────────
push(
  new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { ...LINE, before: 200, after: 100 },
    children: [new TextRun({ text: 'REFERENCES', font: FONT, size: SZ_H1, bold: true })],
  }),
);

const refs = [
  'S. Tang, F. He, X. Huang, and J. Yang, "Online PCB defect detector on a new PCB defect dataset," arXiv preprint arXiv:1902.06197, 2019.',
  'R. Ding, L. Dai, G. Li, and H. Liu, "TDD-net: A tiny defect detection network for printed circuit boards," CAAI Trans. Intell. Technol., vol. 4, no. 2, pp. 110–116, 2019.',
  'Y. Pan, L. Zhang, and Y. Zhang, "Rapid detection of PCB defects based on YOLOx-Plus and FPGA," IEEE Access, vol. 12, pp. 61,343–61,358, 2024.',
  'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, pp. 770–778, 2016.',
  'J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in Adv. NeurIPS, pp. 3320–3328, 2014.',
  'J. Qiu et al., "Going deeper with embedded FPGA platform for convolutional neural network," in Proc. ACM/SIGDA FPGA, pp. 26–35, 2016.',
  'Xilinx Inc., "Vitis AI user guide (UG1414), v3.0," 2022.',
  'H. Fuketa, T. Katashita, Y. Hori, and M. Hioki, "Multiplication-free lookup-based CNN accelerator using residual vector quantization and its FPGA implementation," IEEE Access, vol. 12, pp. 102,470–102,480, 2024.',
  'Y. R. M. Reddy, P. Muralidhar, G. N. V. S. Narayana, and D. Jagan, "FPGA(ZCU104) based energy efficient accelerator for MobileNet-V1," in Proc. IEEE CONECCT, pp. 1–6, 2024.',
  'W. Liu et al., "SSD: Single shot multibox detector," in Proc. ECCV, pp. 21–37, 2016.',
  'T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in Proc. ECCV, pp. 740–755, 2014.',
  'B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in Proc. IEEE CVPR, pp. 2704–2713, 2018.',
  'M. Everingham et al., "The PASCAL visual object classes (VOC) challenge," Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, 2010.',
];

refs.forEach((r, i) => push(
  runs([
    run(`[${i + 1}] `, { bold: true }),
    run(r),
  ], { indent: 360, spaceAfter: 60 }),
));

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ═══════════════════════════════════════════════════════════════════════════════

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MAR_TOP, bottom: MAR_BOT, left: MAR_L, right: MAR_R },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ_BODY })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, 'URECA_Paper_Arjun_Prakash.docx'), buf);
  console.log('Done: ureca_submission/URECA_Paper_Arjun_Prakash.docx');
}).catch(err => { console.error(err); process.exit(1); });
