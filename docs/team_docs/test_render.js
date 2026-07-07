const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');

(async () => {
    await mdToPdf(
        { content: '# Test \n $A_b = C_d$\n $$ \\sum_{i=1}^n X_i $$' },
        {
            dest: 'test_output.pdf',
            launch_options: { args: ['--no-sandbox'] },
            pdf_options: {
                format: 'A4',
                margin: { top: '20mm', bottom: '20mm', left: '20mm', right: '20mm' },
                displayHeaderFooter: true,
                headerTemplate: '<span></span>',
                footerTemplate: '<div style="font-size: 12px; text-align: center; width: 100%;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'
            },
            script: [
                { content: 'window.MathJax = { tex: { inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]] } };' },
                { src: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js' }
            ]
        }
    );
    console.log('PDF generated.');
})();
