document.getElementById('loanForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Collect form data
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    try {
        // Make API call to Flask backend
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok) {
            const prediction = result.prediction;
            const confidence = (result.confidence * 100).toFixed(2);

            // Display result
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `<p>Predicted Loan Status: ${prediction}</p><p>Prediction Confidence: ${confidence}%</p>`;
            resultDiv.style.backgroundColor = prediction === 'Fully Paid' ? '#d4edda' : '#f8d7da';
            resultDiv.style.color = prediction === 'Fully Paid' ? '#155724' : '#721c24';
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
    } catch (error) {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
        resultDiv.style.backgroundColor = '#f8d7da';
        resultDiv.style.color = '#721c24';
    }
});
