const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');

async function generate() {
    const files = [
        '企业级量化因子评估与自动化入库系统需求文档.md',
        '企业级时序因子体系设计与实施总说明.md',
        '反馈意见_对模块间服务接口契约的改进建议.md'
    ];

    for (const file of files) {
        console.log(`Converting ${file}...`);
        await mdToPdf(
            { path: file },
            {
                dest: file.replace('.md', '.pdf'),
                launch_options: { args: ['--no-sandbox', '--disable-setuid-sandbox'] },
                pdf_options: {
                    format: 'A4',
                    margin: { top: '15mm', bottom: '20mm', left: '10mm', right: '10mm' },
                    displayHeaderFooter: true,
                    headerTemplate: '<span></span>',
                    footerTemplate: '<div style="font-size: 10px; width: 100%; text-align: center; color: #666; margin-bottom: 5px;">- <span class="pageNumber"></span> -</div>',
                    printBackground: true
                },
                css: `
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 12px; line-height: 1.5; }
                    h1, h2, h3 { color: #1d4ed8; margin-top: 20px; margin-bottom: 10px; }
                    table { width: 100% !important; border-collapse: collapse; margin-bottom: 15px; table-layout: fixed; }
                    th, td { border: 1px solid #e2e8f0; padding: 6px 8px; font-size: 11px; word-wrap: break-word; overflow-wrap: break-word; }
                    th { background-color: #f8fafc; font-weight: 600; }
                    tr:nth-child(even) { background-color: #fcfcfc; }
                    .mermaid { margin: 20px 0; text-align: center; }
                    code { background-color: #f1f5f9; padding: 2px 4px; border-radius: 4px; font-size: 11px; }
                    pre { background-color: #f1f5f9; padding: 12px; border-radius: 8px; overflow-x: auto; }
                    pre code { background-color: transparent; padding: 0; font-size: 11px; }
                    .math { overflow-x: auto; }
                `,
                script: [
                    { url: 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-MML-AM_CHTML' },
                    { content: 'MathJax.Hub.Config({ tex2jax: { inlineMath: [["$","$"], ["\\\\(","\\\\)"]], displayMath: [["$$","$$"]], processEscapes: true } });' }
                ]
            }
        ).catch(console.error);
    }
    console.log('Done!');
}

generate();
