const { mdToPdf } = require('md-to-pdf');

(async () => {
    await mdToPdf(
        { content: '# Test \n $A_b = C_d$\n $$ \\sum_{i=1}^n X_i $$' },
        {
            dest: 'test_output3.html',
            as_html: true,
            script: [
                { url: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js' }
            ]
        }
    );
    console.log('HTML generated.');
})();
