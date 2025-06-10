// pages/index.tsx (or pages/predict.tsx)
'use client';
import { useState, FormEvent, ChangeEvent } from 'react';

// Define TypeScript interfaces for your data
interface FormData {
  Professional_Status: string;
  Sector: string;
  Existing_Loan: string;
  Total_Acquisition_Price_DT: number;
  Repayment_Duration_Years: number;
  Monthly_Payment_DT: number;
  Documents_Complete: boolean;
  Number_of_Clicks: number;
  Time_Spent_Seconds: number;
}

interface PredictionResult {
  prediction: number;
  probability_not_approved: number;
  probability_approved: number;
}

export default function PredictPage() {
  const [formData, setFormData] = useState<FormData>({
    Professional_Status: 'Salarié',
    Sector: 'Secteur Public',
    Existing_Loan: 'Non',
    Total_Acquisition_Price_DT: 50000,
    Repayment_Duration_Years: 5,
    Monthly_Payment_DT: 1100,
    Documents_Complete: true,
    Number_of_Clicks: 75,
    Time_Spent_Seconds: 400,
  });

  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    // Handle checkbox (boolean) differently
    const newValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked : value;

    setFormData((prevData) => ({
      ...prevData,
      [name]: newValue,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setPredictionResult(null);
    setError(null);

    // Ensure numeric fields are correctly parsed to numbers
    const payload = {
      ...formData,
      Total_Acquisition_Price_DT: parseFloat(formData.Total_Acquisition_Price_DT.toString()),
      Repayment_Duration_Years: parseInt(formData.Repayment_Duration_Years.toString(), 10),
      Monthly_Payment_DT: parseFloat(formData.Monthly_Payment_DT.toString()),
      Number_of_Clicks: parseInt(formData.Number_of_Clicks.toString(), 10),
      Time_Spent_Seconds: parseInt(formData.Time_Spent_Seconds.toString(), 10),
    };

    try {
      const response = await fetch('http://localhost:5000/predict', { // Make sure this URL matches your Flask API
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data: PredictionResult = await response.json();
      setPredictionResult(data);
    } catch (err: any) {
      console.error("Prediction error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500";
  const labelClass = "block text-sm font-medium text-gray-700 mb-1";
  const buttonClass = "w-full py-2 px-4 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-lg shadow-lg max-w-2xl w-full">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">Loan Approval Prediction</h1>
        <p className="text-center text-gray-600 mb-8">Enter the client details to get a prediction from the Flask API.</p>

        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          <div>
            <label htmlFor="Professional_Status" className={labelClass}>Professional Status:</label>
            <select name="Professional_Status" id="Professional_Status" value={formData.Professional_Status} onChange={handleChange} className={inputClass} required>
              <option value="Salarié">Salarié</option>
              <option value="Salarié Et Professionnel À Usage Privé / Rentier">Salarié Et Professionnel À Usage Privé / Rentier</option>
              <option value="Professionnel À Usage Privé">Professionnel À Usage Privé</option>
            </select>
          </div>

          <div>
            <label htmlFor="Sector" className={labelClass}>Sector:</label>
            <select name="Sector" id="Sector" value={formData.Sector} onChange={handleChange} className={inputClass} required>
              <option value="Secteur Privé">Secteur Privé</option>
              <option value="Secteur Public">Secteur Public</option>
            </select>
          </div>

          <div>
            <label htmlFor="Existing_Loan" className={labelClass}>Existing Loan:</label>
            <select name="Existing_Loan" id="Existing_Loan" value={formData.Existing_Loan} onChange={handleChange} className={inputClass} required>
              <option value="Non">Non</option>
              <option value="Oui">Oui</option>
            </select>
          </div>

          <div>
            <label htmlFor="Total_Acquisition_Price_DT" className={labelClass}>Total Acquisition Price (DT):</label>
            <input type="number" name="Total_Acquisition_Price_DT" id="Total_Acquisition_Price_DT" value={formData.Total_Acquisition_Price_DT} onChange={handleChange} className={inputClass} required step="any" />
          </div>

          <div>
            <label htmlFor="Repayment_Duration_Years" className={labelClass}>Repayment Duration (Years):</label>
            <input type="number" name="Repayment_Duration_Years" id="Repayment_Duration_Years" value={formData.Repayment_Duration_Years} onChange={handleChange} className={inputClass} required />
          </div>

          <div>
            <label htmlFor="Monthly_Payment_DT" className={labelClass}>Monthly Payment (DT):</label>
            <input type="number" name="Monthly_Payment_DT" id="Monthly_Payment_DT" value={formData.Monthly_Payment_DT} onChange={handleChange} className={inputClass} required step="any" />
          </div>

          <div className="md:col-span-2 flex items-center mt-2">
            <input type="checkbox" name="Documents_Complete" id="Documents_Complete" checked={formData.Documents_Complete} onChange={handleChange} className="mr-2 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500" />
            <label htmlFor="Documents_Complete" className="text-sm font-medium text-gray-700">Documents Complete</label>
          </div>

          <div>
            <label htmlFor="Number_of_Clicks" className={labelClass}>Number of Clicks:</label>
            <input type="number" name="Number_of_Clicks" id="Number_of_Clicks" value={formData.Number_of_Clicks} onChange={handleChange} className={inputClass} required />
          </div>

          <div>
            <label htmlFor="Time_Spent_Seconds" className={labelClass}>Time Spent (Seconds):</label>
            <input type="number" name="Time_Spent_Seconds" id="Time_Spent_Seconds" value={formData.Time_Spent_Seconds} onChange={handleChange} className={inputClass} required />
          </div>

          <div className="md:col-span-2 text-center mt-4">
            <button type="submit" disabled={loading} className={buttonClass}>
              {loading ? 'Predicting...' : 'Get Prediction'}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-md text-center">
            <p>Error: {error}</p>
          </div>
        )}

        {predictionResult && (
          <div className="mt-6 p-6 bg-blue-50 border border-blue-400 text-blue-800 rounded-lg shadow-sm">
            <h2 className="text-xl font-bold text-blue-700 mb-3">Prediction Result:</h2>
            <p className="mb-2">
              <strong className="text-blue-900">Approval Status:</strong>{' '}
              <span className={`font-semibold ${predictionResult.prediction === 1 ? 'text-green-600' : 'text-red-600'}`}>
                {predictionResult.prediction === 1 ? 'Approved' : 'Not Approved'}
              </span>
            </p>
            <p className="mb-2">
              <strong className="text-blue-900">Probability of Approval:</strong>{' '}
              <span className="font-medium">{(predictionResult.probability_approved * 100).toFixed(2)}%</span>
            </p>
            <p>
              <strong className="text-blue-900">Probability of Not Approved:</strong>{' '}
              <span className="font-medium">{(predictionResult.probability_not_approved * 100).toFixed(2)}%</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}