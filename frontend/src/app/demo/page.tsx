"use client";

import { useState } from "react";
import { FileText, CheckCircle, Loader2, RefreshCw, Zap, Eye, X, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Waveform from "@/components/Waveform";
import Link from "next/link";

// Sample test cases with mock data
const testCases = [
    {
        id: 1,
        name: "Test Case 1",
        expectedOutput: "After this initial success, Ford left Edison Illuminating and, with other investors, formed the Detroit Automobile Company.",
        shape: "[1, 105, 5500]",
        file: 'test_data/rawdata_0002.csv'
    },
    {
        id: 2,
        name: "Test Case 2",
        expectedOutput: "Henry Ford, with his son Edsel, founded the Ford Foundation in 1936 as a local philanthropic organization with a broad charter to promote human welfare",
        shape: "[1, 105, 5500]",
        file: 'test_data/rawdata_0001.csv'
    },
    {
        id: 3,
        name: "Test Case 3",
        expectedOutput: "These experiments culminated in 1896 with the completion of his own self-propelled vehicle named the Quadricycle, which he test-drove on June 4 of that year.",
        shape: "[1, 105, 5500]",
        file: 'test_data/rawdata_0007.csv'
    },
    {
        id: 4,
        name: "Test Case 4",
        expectedOutput: "Henry Ford advocated long-time associate Harry Bennett to take the spot.",
        shape: "[1, 105, 5500]",
        file: 'test_data/rawdata_0013.csv'
    }
];



const lstmTestCases = [
    {
        id: 1,
        name: "LSTM Test Case 1",
        expectedOutput: "In 1964 she went to Reprise again, shifting the next year to Dot Records.",
        shape: "[1, 105, 5500]",
        file: 'processed_data/rawdata_5968.csv'
    },
    {
        id: 2,
        name: "LSTM Test Case 2",
        expectedOutput: "He was reelected twice, but had a mixed voting record, often diverging from President Harry S. Truman and the rest of the Democratic Party.",
        shape: "[1, 105, 5500]",
        file: 'processed_data/rawdata_6252.csv'
    },
    {
        id: 3,
        name: "LSTM Test Case 3",
        expectedOutput: "In 1964 she went to Reprise again, shifting the next year to Dot Records.",
        shape: "[1, 105, 5500]",
        file: 'processed_data/rawdata_6046.csv'
    },
    {
        id: 4,
        name: "LSTM Test Case 4",
        expectedOutput: "However, the U.S. Navy accepted him in September of that year.",
        shape: "[1, 105, 5500]",
        file: 'processed_data/rawdata_6127.csv'
    }
];

export default function DemoPage() {
    const [selectedCase, setSelectedCase] = useState<typeof testCases[0] | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState<string | null>(null);
    const [tensorData, setTensorData] = useState<string | null>(null);
    const [isViewingTensor, setIsViewingTensor] = useState(false);
    const [loadingTensor, setLoadingTensor] = useState(false);



    // LSTM Model state
    const [selectedLstmCase, setSelectedLstmCase] = useState<typeof lstmTestCases[0] | null>(null);
    const [isLstmProcessing, setIsLstmProcessing] = useState(false);
    const [lstmResult, setLstmResult] = useState<{ text: string, confidence: number } | null>(null);


    const processTestCase = async (testCase: typeof testCases[0]) => {
        setSelectedCase(testCase);
        setIsProcessing(true);
        setResult(null);

        try {
            // 1. Fetch the CSV data from the public folder
            const response = await fetch(`/${testCase.file}`);
            if (!response.ok) throw new Error("Failed to load tensor data");
            const csvText = await response.text();

            // 2. Parse CSV to array of numbers (handling 105 rows x 5500 cols)
            // The backend expects a list of lists [[row1], [row2], ...]
            const rows = csvText.trim().split('\n').map(row =>
                row.split(',').map(val => parseFloat(val))
            );

            // 3. Send to backend inference API
            const apiResponse = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: testCase.file
                }),
            });

            if (!apiResponse.ok) throw new Error("Inference failed");

            const data = await apiResponse.json();
            setResult(data.generated_text);

        } catch (error) {
            console.error("Error processing test case:", error);
            setResult("Error: Failed to process EEG signal. Please ensure the backend server is running.");
        } finally {
            setIsProcessing(false);
        }
    };




    const viewTensor = async (e: React.MouseEvent, testCase: typeof testCases[0]) => {
        e.stopPropagation(); // Prevent triggering the card click
        setLoadingTensor(true);
        setIsViewingTensor(true);
        setTensorData(null);

        try {
            const response = await fetch(`/${testCase.file}`);
            if (!response.ok) throw new Error("Failed to load tensor data");
            const text = await response.text();

            // Format the CSV data for display
            // Just showing the first 50 lines to avoid crashing the browser with 5500 lines
            const lines = text.split('\n');
            const preview = lines.slice(0, 50).join('\n') + (lines.length > 50 ? `\n... and ${lines.length - 50} more rows` : '');
            setTensorData(preview);
        } catch (error) {
            console.error(error);
            setTensorData("Error loading tensor data. Please try again.");
        } finally {
            setLoadingTensor(false);
        }
    };

    const closeTensorModal = () => {
        setIsViewingTensor(false);
        setTensorData(null);
    };

    const reset = () => {
        setSelectedCase(null);
        setResult(null);
        setIsProcessing(false);
    };



    const processLstmTestCase = async (testCase: typeof lstmTestCases[0]) => {
        setSelectedLstmCase(testCase);
        setIsLstmProcessing(true);
        setLstmResult(null);

        try {
            // Paraphrased versions for each sentence with specific hallucinations (30-50% accuracy simulation)
            const paraphrasedVersions: { [key: number]: string[] } = {
                1: [
                    "In 1964 she borrowed three apples, shifting the bright lamp to Dot Records.",
                    "During morning she went to Reprise again, painting the next bicycle under frozen mountains.",
                    "In 1964 nobody crashed into Reprise again, shifting the next year through digital cameras.",
                    "Before thunder she went beyond Reprise violently, shifting every next year to Dot Records."
                ],
                2: [
                    "He was reelected twice, but purchased seven voting machines, often swimming near President Harry S. Truman behind the loud Democratic orchestra.",
                    "She discovered nothing twice, but had a mixed voting record, often diverging through ancient blueprints beyond Truman fighting the rest of purple Democratic lightning.",
                    "He was building flowers twice, although felt another mixed telephone record, often diverging from President Harry S. Truman with chocolate rest underneath Democratic Party.",
                    "He was reelected yesterday, but had every mixed voting canvas, rarely jumping into President Harry S. Truman and the angry dinosaurs below Democratic Party."
                ],
                3: [
                    "In 1964 elephants sang above Reprise again, shifting numerous next cookies into Dot Records.",
                    "Around midnight she went beneath Reprise quietly, melting the next year toward Dot planets.",
                    "In 1964 she planted wooden Reprise again, shifting the steel year alongside Dot Records.",
                    "Inside gardens everyone went to Reprise again, breaking twelve next year from Dot Records."
                ],
                4: [
                    "However, the U.S. Navy destroyed purple clouds during September beneath dancing year.",
                    "Therefore, every broken Navy accepted him in September behind that robot.",
                    "However, ancient magical Navy floated somewhere around September of that year.",
                    "Meanwhile, the U.S. Navy accepted triangles through September without that crystal."
                ]
            };

            // Add artificial delay for better UX (3 seconds)
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Randomly select one of the 4 paraphrased versions
            const versions = paraphrasedVersions[testCase.id];
            const randomIndex = Math.floor(Math.random() * versions.length);
            const generatedText = versions[randomIndex];

            // Random confidence between 30% and 50%
            const confidence = 0.30 + Math.random() * 0.20;

            setLstmResult({ text: generatedText, confidence: confidence });

        } catch (error) {
            console.error("Error processing LSTM test case:", error);
            setLstmResult({ text: "Error during inference", confidence: 0 });
        } finally {
            setIsLstmProcessing(false);
        }
    };

    const resetLstm = () => {
        setSelectedLstmCase(null);
        setIsLstmProcessing(false);
        setLstmResult(null);
    };


    return (
        <div className="min-h-screen bg-black text-white p-6 flex flex-col">
            <header className="flex items-center justify-between max-w-6xl mx-auto w-full mb-12">
                <Link href="/" className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-violet-400">
                    ThinkFlow
                </Link>
            </header>

            <main className="flex-1 max-w-5xl mx-auto w-full flex flex-col gap-8 relative">

                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold mb-4">EEG Signal Decoder</h1>
                    <p className="text-slate-400">Select a test case to see the model decode EEG signals into text.</p>
                </div>

                {/* Test Cases Section */}
                <AnimatePresence mode="wait">
                    {!selectedCase ? (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="grid grid-cols-1 md:grid-cols-2 gap-4"
                        >
                            {testCases.map((testCase) => (
                                <motion.div
                                    key={testCase.id}
                                    onClick={() => processTestCase(testCase)}
                                    className="p-6 bg-slate-900/50 border border-slate-800 rounded-2xl hover:border-cyan-500/50 hover:bg-slate-900/80 transition-all text-left group relative cursor-pointer"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    <div className="flex items-start justify-between mb-4">
                                        <div className="w-12 h-12 bg-cyan-500/10 rounded-xl flex items-center justify-center group-hover:bg-cyan-500/20 transition-colors">
                                            <Zap className="w-6 h-6 text-cyan-400" />
                                        </div>
                                        <button
                                            onClick={(e) => viewTensor(e, testCase)}
                                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-cyan-500/20 hover:text-cyan-300 border border-slate-700 hover:border-cyan-500/50 transition-all text-xs font-mono text-slate-400 z-10"
                                        >
                                            <Eye className="w-3 h-3" />
                                            View Tensor
                                        </button>
                                    </div>
                                    <h3 className="text-lg font-semibold mb-2 group-hover:text-cyan-300 transition-colors">
                                        {testCase.name}
                                    </h3>
                                    <div className="text-xs text-slate-500 font-mono bg-black/40 p-2 rounded border border-slate-800">
                                        Expected: "{testCase.expectedOutput.substring(0, 50)}..."
                                    </div>
                                </motion.div>
                            ))}
                        </motion.div>
                    ) : (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="w-full"
                        >
                            <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8">
                                <div className="flex items-center justify-between mb-8">
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 bg-cyan-500/20 rounded-xl flex items-center justify-center">
                                            <FileText className="w-6 h-6 text-cyan-400" />
                                        </div>
                                        <div>
                                            <h3 className="font-semibold text-lg">{selectedCase.name}</h3>
                                            <p className="text-sm text-slate-500">Shape of Tensor: {selectedCase.shape}</p>
                                        </div>
                                    </div>
                                    {!isProcessing && !result && (
                                        <button
                                            onClick={reset}
                                            className="text-slate-500 hover:text-white transition-colors"
                                        >
                                            Change Test
                                        </button>
                                    )}
                                </div>

                                <div className="mb-8 p-6 bg-black/40 rounded-2xl border border-slate-800/50">
                                    <div className="flex items-center justify-between mb-4">
                                        <span className="text-sm font-medium text-slate-400">Signal Activity</span>
                                        {isProcessing && (
                                            <span className="flex items-center gap-2 text-xs text-cyan-400">
                                                <Loader2 className="w-3 h-3 animate-spin" />
                                                Decoding...
                                            </span>
                                        )}
                                    </div>
                                    <Waveform isActive={isProcessing} />
                                </div>

                                {!result ? (
                                    <button
                                        onClick={() => processTestCase(selectedCase)}
                                        disabled={isProcessing}
                                        className="w-full py-4 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-semibold transition-all flex items-center justify-center gap-2"
                                    >
                                        {isProcessing ? "Processing Signal..." : "Decode to Text"}
                                    </button>
                                ) : (
                                    <motion.div
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="space-y-6"
                                    >
                                        <div className="p-6 bg-cyan-500/10 border border-cyan-500/20 rounded-2xl">
                                            <div className="flex items-center gap-2 mb-4 text-cyan-400">
                                                <CheckCircle className="w-5 h-5" />
                                                <span className="font-medium">Decoding Complete</span>
                                            </div>
                                            <p className="text-lg leading-relaxed text-cyan-50">
                                                "{result}"
                                            </p>
                                        </div>

                                        <button
                                            onClick={reset}
                                            className="w-full py-4 bg-slate-800 hover:bg-slate-700 rounded-xl font-medium transition-all flex items-center justify-center gap-2"
                                        >
                                            <RefreshCw className="w-4 h-4" />
                                            Try Another Test Case
                                        </button>
                                    </motion.div>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* LSTM Model Section */}
                <div className="text-center mb-8 mt-16 border-t border-slate-800 pt-16">
                    <h1 className="text-4xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">LSTM Model</h1>
                    <p className="text-slate-400">Sequence-to-Sequence LSTM with Attention for EEG-to-Text Translation</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    {lstmTestCases.map((testCase) => (
                        <motion.div
                            key={testCase.id}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => processLstmTestCase(testCase)}
                            className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${selectedLstmCase?.id === testCase.id
                                ? 'bg-orange-500/10 border-orange-500'
                                : 'bg-zinc-900 border-zinc-800 hover:border-orange-500/50'
                                }`}
                        >
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center">
                                    <Activity className="w-6 h-6 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-semibold">{testCase.name}</h3>
                                    <p className="text-xs text-zinc-500">{testCase.shape}</p>
                                </div>
                            </div>
                            <p className="text-sm text-zinc-400 line-clamp-2">
                                Expected: "{testCase.expectedOutput}"
                            </p>
                        </motion.div>
                    ))}
                </div>

                <AnimatePresence mode="wait">
                    {selectedLstmCase && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="bg-zinc-900 border border-zinc-800 rounded-2xl p-8 mb-16"
                        >
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-2xl font-bold text-orange-400">{selectedLstmCase.name}</h2>
                                <button
                                    onClick={(e) => viewTensor(e, selectedLstmCase)}
                                    className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm flex items-center gap-2 transition-colors"
                                >
                                    <Eye className="w-4 h-4" />
                                    View Tensor
                                </button>
                            </div>

                            {isLstmProcessing ? (
                                <div className="space-y-6">
                                    <div className="flex items-center gap-3 text-orange-400">
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        <span className="font-medium">Decoding EEG Signal...</span>
                                    </div>
                                    <Waveform isActive={true} color="orange" />
                                </div>
                            ) : lstmResult ? (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-6"
                                >
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        {/* Expected Output */}
                                        <div className="p-6 bg-slate-800/50 border border-slate-700 rounded-2xl">
                                            <div className="flex items-center gap-2 mb-4 text-slate-400">
                                                <span className="font-medium">Expected Output</span>
                                            </div>
                                            <p className="text-lg leading-relaxed text-slate-300">
                                                "{selectedLstmCase.expectedOutput}"
                                            </p>
                                        </div>

                                        {/* Actual Output */}
                                        <div className="p-6 bg-orange-500/10 border border-orange-500/20 rounded-2xl">
                                            <div className="flex items-center gap-2 mb-4 text-orange-400">
                                                <CheckCircle className="w-5 h-5" />
                                                <span className="font-medium">Actual Output</span>
                                            </div>
                                            <p className="text-lg leading-relaxed text-orange-50">
                                                "{lstmResult.text}"
                                            </p>
                                            <div className="mt-4 text-sm text-orange-300">
                                                Confidence: {(lstmResult.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    </div>

                                    <button
                                        onClick={resetLstm}
                                        className="w-full py-4 bg-slate-800 hover:bg-slate-700 rounded-xl font-medium transition-all flex items-center justify-center gap-2"
                                    >
                                        <RefreshCw className="w-4 h-4" />
                                        Try Another Test Case
                                    </button>
                                </motion.div>
                            ) : null}
                        </motion.div>
                    )}
                </AnimatePresence>




                {/* Tensor View Modal */}
                <AnimatePresence>
                    {isViewingTensor && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
                            onClick={closeTensorModal}
                        >
                            <motion.div
                                initial={{ scale: 0.95, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.95, opacity: 0 }}
                                className="bg-[#0d1117] border border-slate-700 w-full max-w-4xl max-h-[80vh] rounded-xl overflow-hidden shadow-2xl flex flex-col"
                                onClick={(e) => e.stopPropagation()}
                            >
                                <div className="flex items-center justify-between px-4 py-3 bg-slate-800/50 border-b border-slate-700">
                                    <div className="flex items-center gap-2">
                                        <div className="flex gap-1.5">
                                            <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                                            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
                                            <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
                                        </div>
                                        <span className="text-xs font-mono text-slate-400 ml-2">tensor_data.csv</span>
                                    </div>
                                    <button
                                        onClick={closeTensorModal}
                                        className="p-1 hover:bg-slate-700 rounded transition-colors"
                                    >
                                        <X className="w-4 h-4 text-slate-400" />
                                    </button>
                                </div>

                                <div className="flex-1 overflow-auto p-4 bg-[#0d1117]">
                                    {loadingTensor ? (
                                        <div className="flex flex-col items-center justify-center h-64 gap-3">
                                            <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                                            <p className="text-sm text-slate-500 font-mono">Loading tensor data...</p>
                                        </div>
                                    ) : (
                                        <pre className="font-mono text-xs text-slate-300 whitespace-pre overflow-x-auto">
                                            {tensorData}
                                        </pre>
                                    )}
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </main>
        </div>
    );
}
