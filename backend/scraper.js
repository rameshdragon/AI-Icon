const puppeteer = require("puppeteer");

async function scrapeWebsite(url) {
  let browser;

  try {
    browser = await puppeteer.launch({
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.goto(url, { waitUntil: "networkidle2", timeout: 30000 });

    const pageData = await page.evaluate(() => {
      const title = document.title || "";
      const description =
        document.querySelector('meta[name="description"]')?.content || "";
      const keywords =
        document.querySelector('meta[name="keywords"]')?.content || "";
      const mainHeading = document.querySelector("h1")?.innerText || "";
      const visibleText = document.body?.innerText?.substring(0, 2000) || "";

      return {
        title,
        description,
        keywords,
        h1: mainHeading,
        text: visibleText,
      };
    });

    console.log(JSON.stringify(pageData));
  } catch (error) {
    console.error(JSON.stringify({ error: error.message }));
    process.exit(1);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

const url = process.argv[2];

if (!url) {
  console.error(JSON.stringify({ error: "No URL provided" }));
  process.exit(1);
}

scrapeWebsite(url);
