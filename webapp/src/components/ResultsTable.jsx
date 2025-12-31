import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export default function ResultsTable({ results }) {
  const [expandedRow, setExpandedRow] = useState(null);

  if (!results || results.length === 0) {
    return <p className="text-slate-400">No results to display</p>;
  }

  return (
    <div className="overflow-x-auto mt-6">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-600">
            <th className="text-left py-3 px-4 text-slate-300 font-semibold w-12"></th>
            <th className="text-left py-3 px-4 text-slate-300 font-semibold">MACHINE ID</th>
            <th className="text-left py-3 px-4 text-slate-300 font-semibold">FAILURE PROBABILITY</th>
            <th className="text-left py-3 px-4 text-slate-300 font-semibold">RISK SCORE</th>
            <th className="text-left py-3 px-4 text-slate-300 font-semibold">PRIORITY</th>
          </tr>
        </thead>
        <tbody>
          {results.map((row) => (
            <React.Fragment key={row.row_index}>
              <tr 
                className="border-b border-slate-700/50 hover:bg-slate-700/20 cursor-pointer transition-colors"
                onClick={() => setExpandedRow(expandedRow === row.row_index ? null : row.row_index)}
              >
                <td className="py-3 px-4 text-center">
                  {expandedRow === row.row_index ? (
                    <ChevronDown className="w-4 h-4 text-blue-400" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-slate-500" />
                  )}
                </td>
                <td className="py-3 px-4 text-slate-200 font-medium">#{row.row_index}</td>
                <td className="py-3 px-4 text-slate-300">
                  {(row.failure_probability * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div className="w-24 bg-slate-700 rounded-full h-2">
                      <div
                        className={`h-full rounded-full transition-all ${
                          row.risk_score >= 65
                            ? 'bg-red-500'
                            : row.risk_score >= 40
                            ? 'bg-yellow-500'
                            : 'bg-green-500'
                        }`}
                        style={{ width: `${row.risk_score}%` }}
                      ></div>
                    </div>
                    <span className="text-slate-300 font-medium">{row.risk_score}</span>
                  </div>
                </td>
                <td className="py-3 px-4">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      row.maintenance_priority === 'High'
                        ? 'bg-red-500/20 text-red-300'
                        : row.maintenance_priority === 'Medium'
                        ? 'bg-yellow-500/20 text-yellow-300'
                        : 'bg-green-500/20 text-green-300'
                    }`}
                  >
                    {row.maintenance_priority}
                  </span>
                </td>
              </tr>

              {expandedRow === row.row_index && (
                <tr className="bg-slate-900/50 border-b border-slate-600">
                  <td colSpan="5" className="py-4 px-4">
                    <div className="space-y-4">
                      {/* Human-readable Explanations */}
                      <div>
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                          <span className="text-orange-400">💡</span> Machine Diagnosis
                        </h4>
                        {row.explanations && row.explanations.length > 0 ? (
                          <ul className="text-slate-300 text-sm space-y-2 ml-6">
                            {row.explanations.map((explanation, idx) => (
                              <li key={idx} className="flex items-start gap-2 border-l-2 border-orange-500/50 pl-3">
                                <span className="text-orange-300 mt-0.5 min-w-fit">{explanation}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-slate-400 ml-6">No specific diagnosis available</p>
                        )}
                      </div>

                      {/* Why this machine is risky (Technical) */}
                      <div>
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                          <span className="text-red-400">⚠️</span> Top Risk Factors (Technical)
                        </h4>
                        {row.top_risk_factors && row.top_risk_factors.length > 0 ? (
                          <ul className="text-slate-300 text-sm space-y-2 ml-6">
                            {row.top_risk_factors.map((factor, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-blue-400 mt-0.5">→</span>
                                <div>
                                  <span className="font-medium text-slate-200">{factor.feature}</span>
                                  <span className="text-slate-400"> (impact score: {factor.impact_score})</span>
                                </div>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-slate-400 ml-6">No risk factors identified</p>
                        )}
                      </div>

                      {/* Recommended Actions */}
                      <div>
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                          <span className="text-green-400">✓</span> Recommended Actions
                        </h4>
                        {row.recommended_actions && row.recommended_actions.length > 0 ? (
                          <ul className="text-slate-300 text-sm space-y-2 ml-6">
                            {row.recommended_actions.map((action, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-green-400 mt-0.5">•</span>
                                <span className="text-green-300">{action}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-slate-400 ml-6">No actions recommended</p>
                        )}
                      </div>

                      {/* Mapped raw columns used for explanation */}
                      {row.explanation_mapping && (
                        <div>
                          <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                            <span className="text-indigo-400">🔗</span> Mapped Columns
                          </h4>
                          <ul className="text-slate-300 text-sm space-y-2 ml-6">
                            {Object.entries(row.explanation_mapping).map(([key, info]) => (
                              <li key={key} className="flex items-start gap-2">
                                <span className="font-medium text-slate-200 min-w-[90px]">{key}</span>
                                <span className="text-slate-400">{(info && info.column) || '—'} = {(info && String(info.value)) || '—'}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* What happens if we delay */}
                      <div className="bg-slate-800/50 border border-orange-500/30 rounded p-3 mt-3">
                        <p className="text-orange-300 text-sm">
                          <span className="font-semibold">⏱ Delay consequence:</span> Delaying maintenance increases risk of unplanned
                          downtime and equipment damage. Early intervention prevents costly failures.
                        </p>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}