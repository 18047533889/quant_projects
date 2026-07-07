const { mdToPdf } = require('md-to-pdf');
const fs = require('fs');

(async () => {
    try {
        console.log('Starting PDF generation with MathJax support...');
        
        // 注入 MathJax 脚本到 HTML 模板中
        const config = {
            launch_options: { args: ['--no-sandbox', '--disable-setuid-sandbox'] },
            pdf_options: {
                format: 'A4',
                margin: '20mm',
                printBackground: true
            },
            // 使用 MathJax 进行公式渲染
            pre_process_md: (md) => {
                // 可以在这里对 Markdown 进行最后的预处理
                return md;
            },
            // 在页面加载后执行脚本
            scripts: [
                'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'
            ],
            // 等待 MathJax 渲染完成
            script: `
                window.MathJax = {
                    tex2jax: {
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                        processEscapes: true
                    }
                };
            `
        };

        const pdf = await mdToPdf(
            { path: '/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md' },
            config
        );

        if (pdf) {
            fs.writeFileSync('/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.pdf', pdf.content);
            console.log('PDF generated successfully with MathJax support.');
        }
    } catch (err) {
        console.error('Failed to generate PDF:', err);
    }
})();