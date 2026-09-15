import { useState } from 'react';
import axios from 'axios';
import { Activity, Heart, Info, ArrowRight, Salad, Dumbbell, Stethoscope, AlertTriangle, CheckCircle } from 'lucide-react';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    age: 54,
    sex: "Male",
    cp: 2,
    trestbps: 132,
    chol: 246,
    fbs: "No",
    restecg: 1,
    thalach: 140,
    exang: "No",
    oldpeak: 1.2,
    slope: 2,
    ca: 0,
    thal: 2
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    
    try {
      // In production, this would point to the deployed FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError("Failed to connect to the prediction engine. Make sure the FastAPI backend is running on port 8000.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (score) => {
    if (score < 30) return "#10b981"; // success
    if (score < 50) return "#f59e0b"; // warning
    if (score < 75) return "#f97316"; // orange
    return "#ef4444"; // danger
  };

  return (
    <>
      <header style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
          <Heart color="#ef4444" fill="#ef4444" size={40} />
          Heart Health AI
        </h1>
        <p>Premium Medical Analytics & Personalized Action Plans</p>
      </header>

      <div className="app-container">
        
        {/* Left Column: Input Form */}
        <div className="glass-panel">
          <h3><Activity size={20} style={{ verticalAlign: 'middle', marginRight: '8px' }}/> Your Health Profile</h3>
          <p>Enter your vitals below to generate a tailored report.</p>
          
          <form onSubmit={handlePredict}>
            <div className="input-group">
              <label>Age</label>
              <input type="number" name="age" value={formData.age} onChange={handleChange} min="20" max="100" />
            </div>
            
            <div className="input-group">
              <label>Biological Sex</label>
              <select name="sex" value={formData.sex} onChange={handleChange}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>

            <div className="input-group">
              <label>Resting Blood Pressure (mmHg)</label>
              <input type="number" name="trestbps" value={formData.trestbps} onChange={handleChange} min="80" max="220" />
            </div>

            <div className="input-group">
              <label>Serum Cholesterol (mg/dl)</label>
              <input type="number" name="chol" value={formData.chol} onChange={handleChange} min="100" max="600" />
            </div>

            <div className="input-group">
              <label>Maximum Heart Rate</label>
              <input type="number" name="thalach" value={formData.thalach} onChange={handleChange} min="60" max="220" />
            </div>
            
            <div className="input-group">
              <label>Exercise Induced Angina?</label>
              <select name="exang" value={formData.exang} onChange={handleChange}>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            
            <div className="input-group">
              <label>Fasting Blood Sugar > 120 mg/dl?</label>
              <select name="fbs" value={formData.fbs} onChange={handleChange}>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <button type="submit" className="btn" disabled={loading}>
              {loading ? "Analyzing..." : "Generate My Plan"} <ArrowRight size={18} />
            </button>
            {error && <p style={{ color: 'var(--danger)', marginTop: '1rem', fontSize: '0.875rem' }}>{error}</p>}
          </form>
        </div>

        {/* Right Column: Dashboard */}
        <div className="glass-panel">
          {!result ? (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
              <Info size={48} style={{ marginBottom: '16px' }} />
              <h2>Waiting for data...</h2>
              <p>Fill out the form and generate your plan to see the results.</p>
            </div>
          ) : (
            <div>
              <div className="metric-box" style={{ borderColor: getRiskColor(result.risk_score) }}>
                <div className="metric-value">{result.risk_score.toFixed(1)}%</div>
                <div className="metric-label">Risk Score</div>
              </div>
              
              <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                {result.risk_score < 30 && <h2 style={{ color: 'var(--success)' }}>Optimal Health</h2>}
                {result.risk_score >= 30 && result.risk_score < 50 && <h2 style={{ color: 'var(--warning)' }}>Mild Risk</h2>}
                {result.risk_score >= 50 && result.risk_score < 75 && <h2 style={{ color: '#f97316' }}>Elevated Risk</h2>}
                {result.risk_score >= 75 && <h2 style={{ color: 'var(--danger)' }}>High Priority</h2>}
                <p>This is an AI-generated educational estimate, not a medical diagnosis.</p>
              </div>

              <h3>Your Weekly AI Action Plan</h3>
              
              <div className="advice-card info">
                <Salad color="var(--primary)" size={24} style={{ flexShrink: 0 }} />
                <div>
                  <h4>Diet & Nutrition</h4>
                  <p>{result.advice.nutrition}</p>
                </div>
              </div>

              <div className="advice-card warning">
                <Dumbbell color="var(--warning)" size={24} style={{ flexShrink: 0 }} />
                <div>
                  <h4>Lifestyle & Movement</h4>
                  <p>{result.advice.lifestyle}</p>
                </div>
              </div>

              <div className="advice-card danger">
                <Stethoscope color="var(--danger)" size={24} style={{ flexShrink: 0 }} />
                <div>
                  <h4>Medical Next Steps</h4>
                  <p>{result.advice.medical}</p>
                </div>
              </div>
              
            </div>
          )}
        </div>

      </div>
    </>
  );
}

export default App;
