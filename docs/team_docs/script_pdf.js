const { mdToPdf } = require('md-to-pdf');

(async () => {
    try {
        const pdf = await mdToPdf(
            { path: '/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md' },
            {
                dest: '/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.pdf',
                launch_options: { args: ['--no-sandbox', '--disable-setuid-sandbox'] },
                pdf_options: {
                    format: 'A4',
                    margin: '20mm'
                }
            }
        );
        console.log('PDF generated successfully');
    } catch (err) {
        console.error('Failed to generate PDF:', err);
    }
})();