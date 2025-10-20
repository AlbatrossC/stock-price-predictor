let priceChart = null;
let rsiChart = null;

// Chart configuration for minimalist design
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 15,
                font: {
                    size: 11,
                    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                }
            }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            cornerRadius: 8,
            titleFont: {
                size: 13
            },
            bodyFont: {
                size: 12
            }
        }
    },
    interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
    }
};

// Company suggestions functionality
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
        const inputValue = stockInput.value.toLowerCase().trim();
        companySuggestions.innerHTML = "";

        if (inputValue) {
            companySuggestions.classList.remove("hidden");
            const filteredCompanies = allCompanies
                .filter(company =>
                    company.name.toLowerCase().includes(inputValue) ||
                    company.symbol.toLowerCase().includes(inputValue)
                )
                .slice(0, 10); // Limit to 10 suggestions

            if (filteredCompanies.length === 0) {
                const noResult = document.createElement("div");
                noResult.className = "company-suggestion-item";
                noResult.textContent = "No matches found";
                noResult.style.cursor = "default";
                noResult.style.color = "#9ca3af";
                companySuggestions.appendChild(noResult);
            } else {
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
            }
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
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59, 130, 246, 0.1)",
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.4,
                },
                {
                    label: "SMA 20",
                    data: data.historical_data.sma20,
                    borderColor: "#f59e0b",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0.4,
                    borderDash: [5, 5],
                },
                {
                    label: "EMA 12",
                    data: data.historical_data.ema12,
                    borderColor: "#10b981",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0.4,
                },
                {
                    label: "EMA 26",
                    data: data.historical_data.ema26,
                    borderColor: "#ec4899",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0.4,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    grid: {
                        display: false,
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0,
                        font: {
                            size: 10
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    },
                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                },
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
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99, 102, 241, 0.1)",
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.4,
                    fill: true,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    grid: {
                        display: false,
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0,
                        font: {
                            size: 10
                        }
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: context => {
                            const value = context.tick.value;
                            if (value === 30 || value === 70) {
                                return "rgba(239, 68, 68, 0.3)";
                            }
                            return "rgba(0, 0, 0, 0.05)";
                        },
                    },
                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                },
            },
            plugins: {
                ...chartDefaults.plugins,
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: 70,
                            yMax: 70,
                            borderColor: 'rgba(239, 68, 68, 0.5)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                        },
                        line2: {
                            type: 'line',
                            yMin: 30,
                            yMax: 30,
                            borderColor: 'rgba(239, 68, 68, 0.5)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                        }
                    }
                }
            }
        },
    });
}

// Form submission handler
document.getElementById("predictionForm").addEventListener("submit", async e => {
    e.preventDefault();
    const form = e.target;
    const results = document.getElementById("results");
    const loadingState = document.getElementById("loadingState");

    results.classList.add("hidden");
    loadingState.classList.remove("hidden");

    // Scroll to loading state on mobile
    if (window.innerWidth < 640) {
        loadingState.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

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
                `text-2xl sm:text-3xl font-bold ${data.prediction === "UP" ? "text-green-600" : "text-red-600"}`;
            document.getElementById("confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById("accuracy").textContent = `${(data.accuracy * 100).toFixed(1)}%`;

            destroyCharts();
            createPriceChart(data);
            createRSIChart(data);

            results.classList.remove("hidden");
            
            // Scroll to results on mobile
            if (window.innerWidth < 640) {
                setTimeout(() => {
                    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error(error);
        alert("An error occurred while processing your request. Please try again.");
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
    document.body.classList.add('modal-open');
});

closeModal.addEventListener('click', () => {
    aboutUsModal.classList.add('hidden');
    document.body.classList.remove('modal-open');
});

window.addEventListener('click', (e) => {
    if (e.target === aboutUsModal) {
        aboutUsModal.classList.add('hidden');
        document.body.classList.remove('modal-open');
    }
});

// Handle window resize for charts
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (priceChart) priceChart.resize();
        if (rsiChart) rsiChart.resize();
    }, 250);
});