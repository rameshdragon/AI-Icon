import React, { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8000";

const DEFAULT_STYLE_OPTIONS = {
  icon_types: [
    "flat",
    "3d",
    "vector",
    "hand-drawn",
    "minimalist",
    "geometric",
    "abstract",
    "isometric",
  ],
  finishes: ["matte", "glossy", "metallic", "shiny"],
  styles: ["modern", "retro", "cartoon", "realistic", "gradient", "monochrome"],
};

const STYLE_SECTIONS = [
  { key: "icon_types", title: "Icon Types" },
  { key: "finishes", title: "Finishes" },
  { key: "styles", title: "Styles" },
];

function pause(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

function buildRequestData(activeTab, prompt, url, selectedStyles, selectedColor) {
  if (activeTab === "manual") {
    return {
      endpoint: `${API_URL}/api/generate/manual`,
      payload: {
        prompt,
        styles: selectedStyles,
        color: selectedColor,
      },
    };
  }

  return {
    endpoint: `${API_URL}/api/generate/url`,
    payload: {
      url,
      styles: selectedStyles,
      color: selectedColor,
    },
  };
}

function getFormError(activeTab, prompt, url, selectedStyles) {
  if (activeTab === "manual" && !prompt.trim()) {
    return "Please enter a prompt";
  }

  if (activeTab === "url" && !url.trim()) {
    return "Please enter a website URL";
  }

  if (selectedStyles.length === 0) {
    return "Please select at least one style";
  }

  return "";
}

function downloadImage(imageBase64, index) {
  const link = document.createElement("a");
  link.href = `data:image/png;base64,${imageBase64}`;
  link.download = `icon-${index + 1}.png`;
  link.click();
}

function App() {
  const [activeTab, setActiveTab] = useState("manual");
  const [prompt, setPrompt] = useState("");
  const [url, setUrl] = useState("");
  const [selectedColor, setSelectedColor] = useState("#3498db");
  const [selectedStyles, setSelectedStyles] = useState([]);
  const [styleOptions, setStyleOptions] = useState(DEFAULT_STYLE_OPTIONS);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loadStyleOptions();
  }, []);

  async function loadStyleOptions() {
    try {
      const response = await axios.get(`${API_URL}/api/styles`);
      setStyleOptions(response.data);
    } catch (requestError) {
      console.log("Could not load styles from backend, using local defaults instead.");
    }
  }

  function toggleStyle(styleName) {
    if (selectedStyles.includes(styleName)) {
      const updatedStyles = selectedStyles.filter((style) => style !== styleName);
      setSelectedStyles(updatedStyles);
      return;
    }

    setSelectedStyles([...selectedStyles, styleName]);
  }

  async function handleGenerate() {
    const formError = getFormError(activeTab, prompt, url, selectedStyles);

    if (formError) {
      setError(formError);
      return;
    }

    setError("");
    setResults(null);
    setLoading(true);

    try {
      const { endpoint, payload } = buildRequestData(
        activeTab,
        prompt,
        url,
        selectedStyles,
        selectedColor
      );

      if (activeTab === "url") {
        setLoadingStep("Reading your website...");
        await pause(800);
      }

      setLoadingStep("Writing a better prompt...");
      await pause(500);

      setLoadingStep("Generating icon images...");

      const response = await axios.post(endpoint, payload);
      setResults(response.data);
      setLoadingStep("");
    } catch (requestError) {
      const backendMessage = requestError.response?.data?.detail;
      setError(backendMessage || "Something went wrong while generating the icons.");
      console.error("Generate request failed:", requestError);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AI Icon Generator</h1>
        <p>Create logos and icons with hosted Llama and Stable Diffusion models.</p>
      </header>

      <main className="panel">
        <div className="tabs">
          <button
            className={activeTab === "manual" ? "tab active" : "tab"}
            onClick={() => setActiveTab("manual")}
          >
            Manual Prompt
          </button>

          <button
            className={activeTab === "url" ? "tab active" : "tab"}
            onClick={() => setActiveTab("url")}
          >
            Website URL
          </button>
        </div>

        <section className="form-section">
          {activeTab === "manual" ? (
            <div className="input-group">
              <label htmlFor="prompt-input">What do you want to create?</label>
              <input
                id="prompt-input"
                type="text"
                value={prompt}
                placeholder="Example: coffee shop logo"
                onChange={(event) => setPrompt(event.target.value)}
              />
            </div>
          ) : (
            <div className="input-group">
              <label htmlFor="url-input">Website URL</label>
              <input
                id="url-input"
                type="url"
                value={url}
                placeholder="https://example.com"
                onChange={(event) => setUrl(event.target.value)}
              />
            </div>
          )}

          <div className="input-group">
            <label htmlFor="color-input">Main Color</label>
            <div className="color-row">
              <input
                id="color-input"
                className="color-input"
                type="color"
                value={selectedColor}
                onChange={(event) => setSelectedColor(event.target.value)}
              />
              <span className="color-code">{selectedColor}</span>
            </div>
          </div>

          <div className="input-group">
            <label>Choose Styles ({selectedStyles.length} selected)</label>

            {STYLE_SECTIONS.map((section) => (
              <div key={section.key} className="style-section">
                <h3>{section.title}</h3>
                <div className="chip-grid">
                  {styleOptions[section.key].map((styleName) => (
                    <button
                      key={styleName}
                      type="button"
                      className={selectedStyles.includes(styleName) ? "chip selected" : "chip"}
                      onClick={() => toggleStyle(styleName)}
                    >
                      {styleName}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {error && <div className="message error">{error}</div>}

          <button className="generate-button" onClick={handleGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate Icons"}
          </button>
        </section>
      </main>

      {loading && (
        <section className="panel loading-panel">
          <div className="spinner" />
          <p className="loading-title">Please wait while the app creates your images.</p>
          <p className="loading-step">{loadingStep}</p>
        </section>
      )}

      {results && !loading && (
        <section className="panel">
          <div className="results-header">
            <div>
              <h2>Your Icons</h2>
              <p className="results-subtitle">Prompt used by the model</p>
            </div>
            <div className="prompt-box" title={results.optimized_prompt}>
              {results.optimized_prompt}
            </div>
          </div>

          <div className="image-grid">
            {results.images.map((image, index) => (
              <article key={index} className="image-card">
                <img src={`data:image/png;base64,${image}`} alt={`Generated icon ${index + 1}`} />
                <button onClick={() => downloadImage(image, index)}>Download PNG</button>
              </article>
            ))}
          </div>
        </section>
      )}

      <footer className="footer">
        <p>This project uses FastAPI, React, Puppeteer, Llama, and Stable Diffusion.</p>
      </footer>
    </div>
  );
}

export default App;
