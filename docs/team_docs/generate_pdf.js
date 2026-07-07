const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const mk = require('markdown-it-katex');
const puppeteer = require('puppeteer');

(async () => {
    try {
        const mdFilePath = path.join(__dirname, '企业级量化因子评估与自动化入库系统需求文档.md');
        const pdfFilePath = path.join(__dirname, '企业级量化因子评估与自动化入库系统需求文档.pdf');

        const mdContent = fs.readFileSync(mdFilePath, 'utf-8');

        // Initialize markdown-it with KaTeX support
        const md = new MarkdownIt({
            html: true,
            breaks: true,
            linkify: true,
            typographer: true
        }).use(mk);

        const htmlContent = md.render(mdContent);

        // Wrap in HTML template with KaTeX CSS and basic styling
        const fullHtml = `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>企业级量化因子评估与自动化入库系统需求文档</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    line-height: 1.5;
                    padding: 0;
                    color: #333;
                    max-width: 100%;
                    margin: 0;
                    font-size: 12px;
                }
                h1, h2, h3, h4, h5, h6 {
                    margin-top: 16px;
                    margin-bottom: 10px;
                    font-weight: 600;
                    line-height: 1.25;
                }
                h1 { font-size: 1.8em; padding-bottom: .3em; border-bottom: 1px solid #eaecef; }
                h2 { font-size: 1.4em; padding-bottom: .3em; border-bottom: 1px solid #eaecef; }
                h3 { font-size: 1.2em; }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 12px;
                    font-size: 10.5px;
                    table-layout: auto;
                }
                table th, table td {
                    padding: 4px 6px;
                    border: 1px solid #999;
                    vertical-align: middle;
                    line-height: 1.3;
                    word-wrap: break-word;
                    word-break: break-word;
                }
                table tr:nth-child(2n) {
                    background-color: #f6f8fa;
                }
                pre {
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 10px;
                    margin-bottom: 10px;
                    overflow: auto;
                }
                code {
                    background-color: rgba(27,31,35,.05);
                    border-radius: 3px;
                    padding: .2em .4em;
                    font-family: SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace;
                    font-size: 90%;
                    white-space: pre-wrap;
                    word-break: break-word;
                }
                pre code {
                    background-color: transparent;
                    padding: 0;
                }
                blockquote {
                    padding: 0 1em;
                    color: #6a737d;
                    border-left: .25em solid #dfe2e5;
                    margin: 0 0 10px 0;
                }
                p, ul, ol {
                    margin-top: 0;
                    margin-bottom: 10px;
                }
            </style>
        </head>
        <body>
            ${htmlContent}
        </body>
        </html>
        `;

        // Launch puppeteer
        const browser = await puppeteer.launch({
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        const page = await browser.newPage();
        
        await page.setContent(fullHtml, { waitUntil: 'networkidle0' });
        
        await page.pdf({
            path: pdfFilePath,
            format: 'A4',
            margin: { top: '10mm', right: '10mm', bottom: '10mm', left: '10mm' },
            printBackground: true,
            displayHeaderFooter: true,
            headerTemplate: '<div></div>',
            footerTemplate: '<div style="font-size: 10px; width: 100%; text-align: center;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'
        });

        await browser.close();
        console.log('PDF generated successfully with markdown-it-katex!');

    } catch (err) {
        console.error('Error:', err);
    }
})();
