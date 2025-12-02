"use client";

import { useState, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import Slide from "@/components/Slide";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { X, Brain, Activity, MessageSquare, ArrowRight } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';

const chartData = [
    { name: 'Transformer', score: 40, color: '#06b6d4' },
    { name: 'LSTM', score: 33.4, color: '#fb923c' }
];

const slides = [
    {
        title: "Problem Statement",
        content: (
            <div className="h-full flex flex-col justify-center">
                {/* Brain-to-Text Illustration */}
                <div className="flex items-center justify-center gap-8 mb-12" >
                    <div className="flex flex-col items-center gap-3">
                        <div className="w-24 h-24 rounded-2xl bg-slate-900/80 border border-slate-700 flex items-center justify-center shadow-[0_0_30px_rgba(139,92,246,0.15)] group hover:border-violet-500/50 transition-all">
                            <Brain className="w-10 h-10 text-violet-400 group-hover:scale-110 transition-transform duration-300" />
                        </div>
                        <span className="text-sm text-slate-400 font-mono tracking-wider">NEURAL SIGNALS</span>
                    </div>

                    <div className="flex flex-col gap-1">
                        <ArrowRight className="w-6 h-6 text-slate-700 animate-pulse" />
                    </div>

                    <div className="flex flex-col items-center gap-3">
                        <div className="w-24 h-24 rounded-2xl bg-slate-900/80 border border-slate-700 flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.15)] group hover:border-cyan-500/50 transition-all">
                            <Activity className="w-10 h-10 text-cyan-400 group-hover:scale-110 transition-transform duration-300" />
                        </div>
                        <span className="text-sm text-slate-400 font-mono tracking-wider">DECODING</span>
                    </div>

                    <div className="flex flex-col gap-1">
                        <ArrowRight className="w-6 h-6 text-slate-700 animate-pulse" />
                    </div>

                    <div className="flex flex-col items-center gap-3">
                        <div className="w-24 h-24 rounded-2xl bg-slate-900/80 border border-slate-700 flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.15)] group hover:border-emerald-500/50 transition-all">
                            <MessageSquare className="w-10 h-10 text-emerald-400 group-hover:scale-110 transition-transform duration-300" />
                        </div>
                        <span className="text-sm text-slate-400 font-mono tracking-wider">COMMUNICATION</span>
                    </div>
                </div >

                <div className="space-y-6 text-lg text-zinc-300 max-w-4xl mx-auto text-center">
                    <p className="leading-relaxed">
                        Communication is a fundamental human right, yet millions of people suffering from <span className="text-slate-100 font-medium">motor neuron diseases</span>, spinal cord injuries, or locked-in syndrome lose the ability to speak or type.
                    </p>
                    <p className="leading-relaxed">
                        Current solutions like eye-tracking or invasive implants are often expensive, slow, or require risky surgery.
                    </p>
                    <div className="p-6 bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/20 rounded-xl mt-8 text-left">
                        <h3 className="text-lg font-semibold text-red-400 mb-2 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>
                            The Challenge
                        </h3>
                        <p className="text-base text-red-100/80">
                            How can we create a <span className="text-white font-medium">non-invasive</span>, <span className="text-white font-medium">affordable</span>, and <span className="text-white font-medium">real-time</span> communication interface that translates brain activity directly into text?
                        </p>
                    </div>
                </div>
            </div >
        ),
    },
    {
        title: "Method 1 : Transformer based approach",
        content: (
            <div className="space-y-4 h-full">
                <div className="text-center mb-4">
                    <h3 className="text-xl font-bold text-cyan-400">EEG-to-Text Representation Learning</h3>
                    <p className="text-sm text-violet-400">Using VQ Encoding and Contrastive Alignment</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300">
                    {/* Left Column: EEG Pipeline */}
                    <div className="space-y-4">
                        <Section title="1. Data Preprocessing">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li>Input: Raw CSV (105 channels × variable timestamps)</li>
                                <li>Zero-padding to 5500 timestamps</li>
                                <li>Final Shape: <code className="text-cyan-300">[Batch, 105, 5500]</code></li>
                            </ul>
                        </Section>

                        <Section title="2. EEG Feature Extraction">
                            <div className="space-y-1 text-xs">
                                <p><strong className="text-white">2.1 Conv Encoder:</strong> 1D CNN → <code className="text-cyan-300">[Batch, 57, 512]</code></p>
                                <p><strong className="text-white">2.2 Self-Attention:</strong> Multi-head attention.</p>
                            </div>
                        </Section>

                        <Section title="3. Vector Quantization (VQ)">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li><strong>Codebook:</strong> 2048 × 512 embeddings</li>
                                <li><strong>Quantization:</strong> Nearest-neighbor</li>
                                <li><strong>Loss:</strong> Codebook + Commitment</li>
                            </ul>
                        </Section>

                        <Section title="4. EEG Reconstruction">
                            <p className="text-xs">Decoder reconstructs waveform from quantized tokens via MSE loss.</p>
                        </Section>
                    </div>

                    {/* Right Column: Text & Alignment */}
                    <div className="space-y-4">
                        <Section title="5. Text Embedding (Word2Vec)">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li>Pretrained Word2Vec (300D) → 512D</li>
                                <li>Interpolated to match EEG length (57)</li>
                            </ul>
                        </Section>

                        <Section title="6. Contrastive Alignment">
                            <p className="text-xs"><strong>NT-Xent Loss:</strong> Aligns EEG and Text embeddings.</p>
                            <p className="mt-1 font-mono text-[10px] bg-slate-800 p-1.5 rounded">
                                L_contrast = -log( exp(sim/τ) / Σ exp(sim_k/τ) )
                            </p>
                        </Section>

                        <Section title="7. Training Objective">
                            <div className="p-2 bg-slate-800/50 rounded-lg border border-slate-700">
                                <p className="font-mono text-center text-cyan-300 text-xs">L_total = L_recon + L_vq + α * L_contrast</p>
                            </div>
                        </Section>

                        <Section title="8. Pipeline Summary">
                            <p className="text-xs">Joint optimization to learn discrete neural representations aligned with language.</p>
                        </Section>
                    </div>
                </div>
            </div>
        ),
    },
    {
        title: "Method 2 : LSTM based approach",
        content: (
            <div className="space-y-4 pr-2">
                <div className="text-center mb-4">
                    <h3 className="text-xl font-bold text-orange-400">Seq2Seq LSTM Architecture</h3>
                    <p className="text-sm text-slate-400">Bidirectional Encoder-Decoder with Attention Mechanism</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300">
                    {/* Left Column */}
                    <div className="space-y-4">
                        <Section title="1. Data Preprocessing">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li>Input: Raw EEG CSV (105 channels)</li>
                                <li><strong>Padding:</strong> Zero-pad to 5500 timestamps</li>
                                <li><strong>Downsampling (2x):</strong> Reduces to 2750 steps</li>
                                <li><strong>Augmentation (6x):</strong> Noise, Shift, Dropout</li>
                            </ul>
                        </Section>

                        <Section title="2. Encoder (Bidirectional LSTM)">
                            <div className="space-y-1 text-xs">
                                <p><strong className="text-white">Architecture:</strong></p>
                                <ul className="list-disc list-inside pl-2 text-slate-400">
                                    <li>Input: 105 EEG channels</li>
                                    <li>2 Layers, 256 Hidden Units</li>
                                    <li>Bidirectional (Forward + Backward)</li>
                                </ul>
                                <p className="mt-1">Output: <code className="text-orange-300">[Batch, 2750, 512]</code></p>
                            </div>
                        </Section>

                        <Section title="3. Attention Mechanism">
                            <p className="text-xs"><strong>Bahdanau Attention:</strong></p>
                            <p className="text-[10px] text-slate-500 mt-1">Learns alignment between EEG timesteps and words.</p>
                            <p className="font-mono text-[10px] bg-slate-800 p-1.5 rounded mt-1">
                                context = sum(attention_weights * encoder_outputs)
                            </p>
                        </Section>


                    </div>

                    {/* Right Column */}
                    <div className="space-y-4">
                        <Section title="4. Decoder (Unidirectional LSTM)">
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-xs">
                                <li><strong>Type:</strong> 2-Layer LSTM (256 units)</li>
                                <li><strong>Input:</strong> Previous word embedding + Context</li>
                                <li><strong>Generation:</strong> Auto-regressive (word-by-word)</li>

                            </ul>
                        </Section>

                        <Section title="5. Training Objectives">
                            <div className="space-y-2">
                                <div>
                                    <strong className="text-white">Teacher Forcing:</strong>
                                    <p className="text-[10px] text-slate-500">Feed ground-truth words during training.</p>
                                </div>
                                <div>
                                    <strong className="text-white">Loss Function:</strong>
                                    <p className="font-mono text-[10px] bg-slate-800 p-1 rounded">CrossEntropy(predicted_logits, true_word)</p>
                                </div>
                                <div>
                                    <strong className="text-white">Optimization:</strong>
                                    <p className="text-[10px] text-slate-500">Adam (lr=0.001) + ReduceLROnPlateau</p>
                                </div>
                            </div>
                        </Section>

                        <Section title="6. Inference Pipeline">
                            <ol className="list-decimal list-inside space-y-0.5 text-slate-400 text-xs">
                                <li>Encode full EEG sequence</li>
                                <li>Initialize decoder with &lt;SOS&gt;</li>
                                <li>Loop until &lt;EOS&gt; or max length:</li>
                                <ul className="list-disc list-inside pl-4 text-[10px]">
                                    <li>Calculate Attention</li>
                                    <li>Predict next word</li>
                                    <li>Feed back as input</li>
                                </ul>
                            </ol>
                        </Section>
                    </div>
                </div>
            </div>
        ),
    },
    {
        title: "Experimentation & Results",
        content: (
            <div className="space-y-6">
                {/* Dataset Information */}
                <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                    <h3 className="text-lg font-semibold mb-3 text-emerald-400">Dataset: ZuCo 2.0</h3>
                    <div className="grid grid-cols-4 gap-4 text-sm">
                        <div className="text-center">
                            <div className="text-2xl font-bold text-cyan-400">5,915</div>
                            <div className="text-slate-400">Total Samples</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-violet-400">80%</div>
                            <div className="text-slate-400">Training Set</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-orange-400">10%</div>
                            <div className="text-slate-400">Validation Set</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-orange-400">10%</div>
                            <div className="text-slate-400">Test Set</div>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Comparison Chart + Key Findings */}
                    <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-800 flex flex-col">
                        <h3 className="text-xl font-semibold mb-6 text-violet-400">BLEU-1 Score Comparison</h3>
                        <div className="w-full h-64 flex items-center justify-center mb-6">
                            <BarChart width={400} height={240} data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="name" stroke="#94a3b8" />
                                <YAxis stroke="#94a3b8" />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #475569',
                                        borderRadius: '8px'
                                    }}
                                    labelStyle={{ color: '#e2e8f0' }}
                                />
                                <Bar dataKey="score" radius={[8, 8, 0, 0]}>
                                    {chartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </div>

                        {/* Key Findings */}
                        <div className="border-t border-slate-700 pt-6 mt-2">
                            <h3 className="text-lg font-semibold mb-4 text-emerald-400">Key Findings</h3>
                            <div className="flex flex-col gap-4 text-sm text-slate-400">
                                <div className="flex items-start gap-3">
                                    <div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-cyan-500 flex-shrink-0" />
                                    <p className="leading-relaxed">The transformer model successfully extracts some semantic information from raw EEG, but still faces significant challenges due to noise, variability, and limited training data.</p>
                                </div>
                                <div className="flex items-start gap-3">
                                    <div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-violet-500 flex-shrink-0" />
                                    <p className="leading-relaxed">The LSTM model effectively captures temporal dependencies in neural activity during natural reading, while the attention mechanism learns interpretable alignments between brain signals and generated words, revealing which neural patterns correspond to specific linguistic elements. </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Training Curves */}
                    <div className="p-6 bg-slate-900/50 rounded-2xl border border-slate-800">
                        <h3 className="text-xl font-semibold mb-4 text-violet-400">Training Curves</h3>
                        <div className="space-y-4">
                            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                                <h4 className="text-sm font-semibold text-cyan-400 mb-2">Transformer Loss</h4>
                                <img
                                    src="/results_images/filtered.png"
                                    alt="Transformer Loss vs Epoch"
                                    className="w-full h-auto rounded border border-slate-700"
                                />
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                                <h4 className="text-sm font-semibold text-orange-400 mb-2">LSTM Loss</h4>
                                <img
                                    src="/results_images/lstm_loss.jpg"
                                    alt="LSTM Loss vs Epoch"
                                    className="w-full h-auto rounded border border-slate-700"
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        ),
    },
    {
        title: "References",
        content: (
            <div className="space-y-6">
                <div className="grid grid-cols-1 gap-4">
                    <ReferenceCard
                        title="DeWave: Discrete EEG waves encoding for brain dynamics to text translation"
                        authors="Y. Duan, J. Zhou, Z. Wang, Y.-K. Wang, and C.-T. Lin (2023)"
                        journal="arXiv preprint arXiv:2309.14030"
                        link="https://arxiv.org/abs/2309.14030"
                    />
                    <ReferenceCard
                        title="Attention Is All You Need"
                        authors="A. Vaswani et al. (2017)"
                        journal="arXiv preprint arXiv:1706.03762"
                        link="https://arxiv.org/abs/1706.03762"
                    />
                    <ReferenceCard
                        title="A simple framework for contrastive learning of visual representations"
                        authors="T. Chen, S. Kornblith, M. Norouzi, and G. Hinton (2020)"
                        journal="arXiv preprint arXiv:2002.05709"
                        link="https://arxiv.org/abs/2002.05709"
                    />
                    <ReferenceCard
                        title="Decoding EEG Brain Activity for Multi-Modal Natural Language Processing"
                        authors="N. Hollenstein et al. (2021)"
                        journal="Front. Hum. Neurosci., vol. 15"
                        link="https://doi.org/10.3389/fnhum.2021.659410"
                    />
                    <ReferenceCard
                        title="Large language models reveal the structure of the mental lexicon"
                        authors="E. Fedorenko, S. Bajaj, J. W. H. Stoeckle, and M. Schrimpf (2024)"
                        journal="Nat. Mach. Intell."
                        link="https://doi.org/10.1038/s42256-024-00824-8"
                    />
                </div>
            </div>
        ),
    }
];

export default function PresentationPage() {
    const [currentSlide, setCurrentSlide] = useState(0);
    const router = useRouter();

    const nextSlide = () => {
        if (currentSlide < slides.length - 1) {
            setCurrentSlide(prev => prev + 1);
        }
    };

    const prevSlide = () => {
        if (currentSlide > 0) {
            setCurrentSlide(prev => prev - 1);
        }
    };

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "ArrowRight") nextSlide();
            if (e.key === "ArrowLeft") prevSlide();
            if (e.key === "Escape") router.push("/");
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [currentSlide]);

    return (
        <div className="h-screen w-screen bg-black text-white overflow-hidden relative flex flex-col">
            {/* Close Button */}
            <Link href="/" className="absolute top-6 right-6 z-50 p-2 rounded-full bg-zinc-900 hover:bg-zinc-800 transition-colors">
                <X className="w-6 h-6 text-zinc-400" />
            </Link>

            {/* Progress Bar */}
            <div className="absolute top-0 left-0 w-full h-1 bg-zinc-900">
                <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-300"
                    style={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
                />
            </div>

            <main className="flex-1 flex flex-col relative overflow-hidden">
                <AnimatePresence mode="wait">
                    <Slide
                        key={currentSlide}
                        title={slides[currentSlide].title}
                        onNext={nextSlide}
                        onPrev={prevSlide}
                        isFirst={currentSlide === 0}
                        isLast={currentSlide === slides.length - 1}
                    >
                        {slides[currentSlide].content}
                    </Slide>
                </AnimatePresence>
            </main>
        </div>
    );
}

function Section({ title, children }: { title: string, children: React.ReactNode }) {
    return (
        <div className="relative rounded-lg bg-[#0d1117] border border-slate-700/50 overflow-hidden group hover:border-cyan-500/40 transition-colors">
            {/* Code block header */}
            <div className="flex items-center justify-between px-4 py-2 bg-slate-800/40 border-b border-slate-700/50">
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
                    </div>
                    <span className="text-xs font-mono text-slate-400 ml-2">{title}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">py</span>
            </div>

            {/* Code content */}
            <div className="p-4 font-mono text-slate-300 text-xs leading-relaxed">
                {children}
            </div>
        </div>
    )
}

function ReferenceCard({ title, authors, journal, link }: { title: string, authors: string, journal: string, link: string }) {
    return (
        <a href={link} target="_blank" rel="noopener noreferrer" className="block p-4 bg-slate-900/50 rounded-xl border border-slate-800 hover:border-violet-500/50 hover:bg-slate-800/50 transition-all group">
            <h4 className="text-base font-semibold text-cyan-400 mb-1 group-hover:text-cyan-300">{title}</h4>
            <p className="text-sm text-slate-300 mb-1">{authors}</p>
            <p className="text-xs text-slate-500 italic">{journal}</p>
        </a>
    )
}
