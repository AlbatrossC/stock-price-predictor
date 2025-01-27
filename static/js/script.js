let priceChart = null;
let rsiChart = null;

document.addEventListener("DOMContentLoaded", () => {
    const stockInput = document.getElementById("stockInput");
    const companySuggestions = document.getElementById("companySuggestions");
    let allCompanies = [];

    async function fetchCompanyNames() {
        try {
            const response = await fetch("/static/companies.json");
            allCompanies = await response.json();
        } catch (error) {
            console.error("Error fetching company names:", error);
        }
    }

    fetchCompanyNames();

    stockInput.addEventListener("input", () => {
        const inputValue = stockInput.value.toLowerCase();
        companySuggestions.innerHTML = "";

        if (inputValue) {
            companySuggestions.classList.remove("hidden");
            const filteredCompanies = allCompanies.filter(company => 
                company.name.toLowerCase().includes(inputValue) ||
                company.symbol.toLowerCase().includes(inputValue)
            );

            filteredCompanies.forEach(company => {
                const suggestion = document.createElement("div");
                suggestion.className = "company-suggestion-item";
                suggestion.textContent = `${company.symbol} - ${company.name}`;
                suggestion.addEventListener("click", () => {
                    stockInput.value = company.symbol;
                    companySuggestions.classList.add("hidden");
                });
                companySuggestions.appendChild(suggestion);
            });
        } else {
            companySuggestions.classList.add("hidden");
        }
    });

    // Hide suggestions when clicking outside
    document.addEventListener("click", (e) => {
        if (!stockInput.contains(e.target) && !companySuggestions.contains(e.target)) {
            companySuggestions.classList.add("hidden");
        }
    });
});

function destroyCharts() {
    if (priceChart) {
        priceChart.destroy();
        priceChart = null;
    }
    if (rsiChart) {
        rsiChart.destroy();
        rsiChart = null;
    }
}

function createPriceChart(data) {
    const ctx = document.getElementById("priceChart").getContext("2d");
    priceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.historical_data.dates,
            datasets: [
                {
                    label: "Close Price",
                    data: data.historical_data.prices,
                    borderColor: "rgb(59, 130, 246)",
                    tension: 0.1,
                },
                {
                    label: "SMA 20",
                    data: data.historical_data.sma20,
                    borderColor: "rgb(234, 179, 8)",
                    tension: 0.1,
                },
                {
                    label: "EMA 12",
                    data: data.historical_data.ema12,
                    borderColor: "rgb(16, 185, 129)",
                    tension: 0.1,
                },
                {
                    label: "EMA 26",
                    data: data.historical_data.ema26,
                    borderColor: "rgb(236, 72, 153)",
                    tension: 0.1,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: "Price History and Moving Averages",
                },
            },
            interaction: {
                intersect: false,
                mode: "index",
            },
        },
    });
}

function createRSIChart(data) {
    const ctx = document.getElementById("rsiChart").getContext("2d");
    rsiChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.historical_data.dates,
            datasets: [
                {
                    label: "RSI",
                    data: data.historical_data.rsi,
                    borderColor: "rgb(99, 102, 241)",
                    tension: 0.1,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: "Relative Strength Index (RSI)",
                },
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: context => {
                            if (context.tick.value === 30 || context.tick.value === 70) {
                                return "rgba(255, 0, 0, 0.2)";
                            }
                            return "rgba(0, 0, 0, 0.1)";
                        },
                    },
                },
            },
        },
    });
}

document.getElementById("predictionForm").addEventListener("submit", async e => {
    e.preventDefault();
    const form = e.target;
    const results = document.getElementById("results");
    const loadingState = document.getElementById("loadingState");

    results.classList.add("hidden");
    loadingState.classList.remove("hidden");

    try {
        const formData = new FormData(form);
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (data.success) {
            document.getElementById("prediction").textContent = data.prediction;
            document.getElementById("prediction").className = 
                `text-3xl font-bold ${data.prediction === "UP" ? "text-green-600" : "text-red-600"}`;
            document.getElementById("confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById("accuracy").textContent = `${(data.accuracy * 100).toFixed(1)}%`;

            destroyCharts();
            createPriceChart(data);
            createRSIChart(data);

            results.classList.remove("hidden");
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert("An error occurred while processing your request.");
    } finally {
        loadingState.classList.add("hidden");
    }
});

// Modal functionality
const aboutUsBtn = document.getElementById('aboutUsBtn');
const aboutUsModal = document.getElementById('aboutUsModal');
const closeModal = document.getElementById('closeModal');

aboutUsBtn.addEventListener('click', () => {
    aboutUsModal.classList.remove('hidden');
});

closeModal.addEventListener('click', () => {
    aboutUsModal.classList.add('hidden');
});

window.addEventListener('click', (e) => {
    if (e.target === aboutUsModal) {
        aboutUsModal.classList.add('hidden');
    }
});
