export default function SummaryPanel({ results }) {
  if (!results || results.length === 0) return null;

  const highRisk = results.filter(
    (r) => r.maintenance_priority === "High"
  ).length;
  
  const mediumRisk = results.filter(
    (r) => r.maintenance_priority === "Medium"
  ).length;
  
  const lowRisk = results.filter(
    (r) => r.maintenance_priority === "Low"
  ).length;

  const avgFailureProbability = (
    results.reduce((sum, r) => sum + (r.failure_probability || 0), 0) / results.length
  ).toFixed(2);

  return (
    <div style={{ marginTop: "20px", padding: "15px", border: "1px solid #ddd", borderRadius: "5px" }}>
      <h3>Summary</h3>
      <p><strong>Total Machines:</strong> {results.length}</p>
      <p><strong>High Risk:</strong> {highRisk}</p>
      <p><strong>Medium Risk:</strong> {mediumRisk}</p>
      <p><strong>Low Risk:</strong> {lowRisk}</p>
      <p><strong>Average Failure Probability:</strong> {avgFailureProbability}</p>
    </div>
  );
}