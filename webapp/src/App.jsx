import React, { useState } from 'react';
import { Upload, AlertCircle, CheckCircle, TrendingUp, Activity, Wrench, FileText, Download } from 'lucide-react';

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:5000';

const App = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Uploading to:', `${API_BASE_URL}/upload-csv`);
      
      const response = await fetch(`${API_BASE_URL}/upload-csv`, {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || errorData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      if (!data.results || data.results.length === 0) {
        throw new Error('No results returned from server');
      }
      
      setResults(data.results);
    } catch (err) {
      console.error('Upload error details:', err);
      
      if (err instanceof TypeError && err.message.includes('Failed to fetch')) {
        setError(`Cannot connect to server at ${API_BASE_URL}. Please check:\n1. Backend is running\n2. CORS is configured\n3. URL is correct`);
      } else {
        setError(err.message || 'Failed to process CSV. Please check your file and try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResults(null);
    setError(null);
  };

  const exportResults = () => {
    if (!results) return;
    
    const csv = [
      ['Row Index', 'Failure Probability', 'Risk Score', 'Maintenance Priority'],
      ...results.map(r => [
        r.row_index,
        r.failure_probability,
        r.risk_score,
        r.maintenance_priority
      ])
    ].map(row => row.join(','))
     .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    a.click();
  };

  // Calculate statistics
  const getStats = () => {
    if (!results) return null;

    const high = results.filter(r => r.maintenance_priority === 'High').length;
    const medium = results.filter(r => r.maintenance_priority === 'Medium').length;
    const low = results.filter(r => r.maintenance_priority === 'Low').length;
    const avgProbability = (results.reduce((sum, r) => sum + (r.failure_probability || 0), 0) / results.length * 100).toFixed(1);

    return { high, medium, low, avgProbability, total: results.length };
  };

  const stats = getStats();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-slate-900/50 backdrop-blur-sm border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Activity className="w-8 h-8 text-blue-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Smart Manufacturing GenAI</h1>
              <p className="text-slate-400 text-sm">Predictive Maintenance Dashboard</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Section */}
        {!results && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-8 mb-8">
            <div className="text-center mb-6">
              <h2 className="text-xl font-semibold text-white mb-2">Upload Manufacturing Data</h2>
              <p className="text-slate-400">Upload a CSV file to analyze machine failure risks and maintenance priorities</p>
            </div>

            {/* Drag & Drop Zone */}
            <div
              className={`relative border-2 border-dashed rounded-xl p-12 transition-all ${
                dragActive 
                  ? 'border-blue-500 bg-blue-500/10' 
                  : 'border-slate-600 hover:border-slate-500'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                id="file-upload"
                className="hidden"
                accept=".csv"
                onChange={handleFileChange}
                disabled={loading}
              />
              
              <label htmlFor="file-upload" className="cursor-pointer">
                <div className="flex flex-col items-center gap-4">
                  <div className="p-4 bg-blue-500/10 rounded-full">
                    <Upload className="w-12 h-12 text-blue-400" />
                  </div>
                  <div className="text-center">
                    <p className="text-white font-medium mb-1">
                      {file ? file.name : 'Drop your CSV file here, or click to browse'}
                    </p>
                    <p className="text-slate-400 text-sm">Supports CSV files up to 10MB</p>
                  </div>
                </div>
              </label>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="text-red-200 text-sm whitespace-pre-line">{error}</div>
              </div>
            )}

            {/* Upload Button */}
            {file && (
              <div className="mt-6 flex gap-3 justify-center">
                <button
                  onClick={handleUpload}
                  disabled={loading}
                  className="px-8 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-lg transition-colors flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="w-5 h-5" />
                      Analyze Data
                    </>
                  )}
                </button>
                
                {!loading && (
                  <button
                    onClick={handleReset}
                    className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {results && stats && (
          <>
            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              <StatCard
                icon={<FileText className="w-6 h-6" />}
                label="Total Machines"
                value={stats.total}
                color="blue"
              />
              <StatCard
                icon={<AlertCircle className="w-6 h-6" />}
                label="High Risk"
                value={stats.high}
                color="red"
              />
              <StatCard
                icon={<Wrench className="w-6 h-6" />}
                label="Medium Risk"
                value={stats.medium}
                color="yellow"
              />
              <StatCard
                icon={<CheckCircle className="w-6 h-6" />}
                label="Avg. Failure Risk"
                value={`${stats.avgProbability}%`}
                color="green"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 mb-6">
              <button
                onClick={exportResults}
                className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Export Results
              </button>
              <button
                onClick={handleReset}
                className="px-6 py-2.5 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-lg transition-colors"
              >
                Upload New File
              </button>
            </div>

            {/* Results Table */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
              <div className="p-6 border-b border-slate-700/50">
                <h3 className="text-lg font-semibold text-white">Analysis Results</h3>
                <p className="text-slate-400 text-sm mt-1">Detailed predictions for each machine</p>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-slate-900/50">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                        Machine ID
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                        Failure Probability
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                        Risk Score
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                        Priority
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/50">
                    {results.map((row, idx) => (
                      <tr
                        key={idx}
                        className="hover:bg-slate-700/30 transition-colors cursor-pointer"
                        onClick={() => setSelectedMachine(row)}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                          #{row.row_index}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                          {(row.failure_probability * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                          <div className="flex items-center gap-2">
                            <div className="w-32 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${
                                  row.risk_score >= 75 ? 'bg-red-500' :
                                  row.risk_score >= 40 ? 'bg-yellow-500' :
                                  'bg-green-500'
                                }`}
                                style={{ width: `${row.risk_score}%` }}
                              />
                            </div>
                            <span>{row.risk_score}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <PriorityBadge priority={row.maintenance_priority} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Details Modal */}
            {selectedMachine && (
              <div className="fixed inset-0 z-50 flex items-center justify-center">
                <div className="absolute inset-0 bg-black/60" onClick={() => setSelectedMachine(null)} />

                <div className="relative bg-slate-900 rounded-2xl border border-slate-700/50 max-w-2xl w-full mx-4 p-6 z-10">
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 className="text-lg font-semibold text-white">Machine Details</h2>
                      <p className="text-slate-400 text-sm">ID: #{selectedMachine.row_index}</p>
                    </div>
                    <button
                      onClick={() => setSelectedMachine(null)}
                      className="text-slate-400 hover:text-white"
                      aria-label="Close details"
                    >
                      ✕
                    </button>
                  </div>

                  <div className="mt-4">
                    <h3 className="text-white font-semibold mb-2">Why is this machine risky?</h3>
                    {(
                      (selectedMachine.explanations && selectedMachine.explanations.length > 0 && selectedMachine.explanations) ||
                      (selectedMachine.explanation && selectedMachine.explanation.length > 0 && selectedMachine.explanation)
                    ) ? (
                      <ul className="list-disc ml-5 text-slate-300 space-y-2">
                        {((selectedMachine.explanations && selectedMachine.explanations.length > 0) ? selectedMachine.explanations : selectedMachine.explanation).map((reason, i) => (
                          <li key={i}>{reason}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-slate-400">No explanation available for this machine.</p>
                    )}
                  </div>

                  {/* Mapped raw columns used for explanation */}
                  {selectedMachine.explanation_mapping && (
                    <div className="mt-4">
                      <h4 className="text-white font-semibold mb-2">Mapped Columns</h4>
                      <ul className="text-slate-300 text-sm ml-4 space-y-1">
                        {Object.entries(selectedMachine.explanation_mapping).map(([key, info]) => (
                          <li key={key} className="flex items-start gap-2">
                            <span className="font-medium text-slate-200 mr-2">{key}:</span>
                            <div className="text-slate-400">{info.column || '—'} = {info.value}</div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Decision Intelligence (failure modes) */}
                  {selectedMachine.decision && (
                    <div className="mt-4">
                      <h4 className="text-white font-semibold mb-2">Decision Intelligence</h4>
                      <p className="text-slate-400 text-sm mb-2">Risk level: <span className="text-white font-medium">{selectedMachine.decision.risk_level}</span></p>
                      <p className="text-slate-400 text-sm mb-2">Action required: <span className="font-medium text-white">{selectedMachine.decision.action_required ? 'Yes' : 'No'}</span></p>

                      {selectedMachine.decision.failure_modes && selectedMachine.decision.failure_modes.length > 0 ? (
                        <ul className="text-slate-300 text-sm ml-4 space-y-1">
                          {selectedMachine.decision.failure_modes.map((fm, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="font-medium text-slate-200 mr-2">{fm.mode}</span>
                              <div className="text-slate-400">({fm.confidence}) — {fm.reason}</div>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-slate-400">No failure modes detected</p>
                      )}
                      {selectedMachine.decision.recommended_actions && selectedMachine.decision.recommended_actions.length > 0 && (
                        <div className="mt-3">
                          <h5 className="text-white font-semibold mb-2">Recommended Actions</h5>
                          <ul className="text-slate-300 text-sm ml-4 space-y-1">
                            {selectedMachine.decision.recommended_actions.map((act, j) => (
                              <li key={j} className="flex flex-col">
                                <span className="font-medium text-slate-200">{act.failure_mode}: {act.recommended_action}</span>
                                <span className="text-slate-400 text-xs">Urgency: {act.urgency} • Downtime: {act.estimated_downtime_min} min • Cost: {act.estimated_cost_inr}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="mt-6 flex justify-end">
                    <button
                      onClick={() => setSelectedMachine(null)}
                      className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg"
                    >
                      Close
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
};

// Stat Card Component
const StatCard = ({ icon, label, value, color }) => {
  const colorClasses = {
    blue: 'bg-blue-500/10 text-blue-400',
    red: 'bg-red-500/10 text-red-400',
    yellow: 'bg-yellow-500/10 text-yellow-400',
    green: 'bg-green-500/10 text-green-400',
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
      <div className={`inline-flex p-3 rounded-lg ${colorClasses[color]} mb-4`}>
        {icon}
      </div>
      <p className="text-slate-400 text-sm mb-1">{label}</p>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  );
};

// Priority Badge Component
const PriorityBadge = ({ priority }) => {
  const styles = {
    High: 'bg-red-500/10 text-red-400 border-red-500/50',
    Medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/50',
    Low: 'bg-green-500/10 text-green-400 border-green-500/50',
  };

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${styles[priority]}`}>
      {priority}
    </span>
  );
};

export default App;