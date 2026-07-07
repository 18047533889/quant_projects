const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');

(async () => {
    try {
        console.log('Starting PDF generation with MathJax support...');
        
        const config = {
            launch_options: { args: ['--no-sandbox', '--disable-setuid-sandbox'] },
            pdf_options: {
                format: 'A4',
                margin: '20mm',
                printBackground: true
            },
            // md-to-pdf 支持在 scripts 中传递对象或字符串
            // 修复 script 传递错误
            script: [
                { url: 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML' },
                { content: `
                    window.MathJax = {
                        tex2jax: {
                            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                            processEscapes: true
                        }
                    };
                `}
            ]
        };

        const pdf = await mdToPdf(
            { path: '/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md' },
            config
        );

        if (pdf) {
            fs.writeFileSync('/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.pdf', pdf.content);
            console.log('PDF generated successfully.');
        }
    } catch (err) {
        console.error('Failed to generate PDF:', err);
    }
})();