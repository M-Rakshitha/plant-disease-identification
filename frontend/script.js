// === Filename preview ===
const input = document.getElementById('file');
const nameEl = document.getElementById('filename');
const submitBtn = document.getElementById('submitBtn');
const footerYear = document.getElementById('year');
footerYear.textContent = new Date().getFullYear();

const BACKEND_URL = "http://127.0.0.1:8000/predict_detail";

// Create display elements dynamically
const resultEl = document.createElement('p');
resultEl.id = 'result';
resultEl.style.fontWeight = '600';
resultEl.style.marginTop = '10px';

const errorEl = document.createElement('p');
errorEl.id = 'error';
errorEl.style.color = 'red';
errorEl.style.marginTop = '10px';

document.querySelector('.center').appendChild(resultEl);
document.querySelector('.center').appendChild(errorEl);

// File name preview
input.addEventListener('change', () => {
  nameEl.textContent = input.files?.[0]?.name || 'No file chosen';
  resultEl.textContent = '';
  errorEl.textContent = '';
});

// Submit button click
submitBtn.addEventListener('click', async () => {
  if (!input.files || input.files.length === 0) {
    errorEl.textContent = 'Please choose an image first!';
    return;
  }

  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);

  resultEl.textContent = 'Analyzing...';
  errorEl.textContent = '';

  try {
    const res = await fetch(BACKEND_URL, {
      method: 'POST',
      body: formData
    });
  
    if (!res.ok) throw new Error(`Server responded with ${res.status}`);
  
    const data = await res.json();
  
    // Store the RICH payload and go to results page
    sessionStorage.setItem("plantResult", JSON.stringify(data));
    window.location.href = "./results.html";
  } catch (err) {
    console.error('Error:', err);
    errorEl.textContent = '⚠️ Could not connect to backend or request failed.';
    resultEl.textContent = '';
  }  
});


