REM Windows git-hooks installation script.
@echo off
pip install black
pip install nbstripout
pip install pre-commit
nbstripout --install
git config core.autocrlf true
git config filter.nbstripout.extrakeys 'metadata.celltoolbar metadata.kernel_spec.display_name metadata.kernel_spec.name metadata.language_info.codemirror_mode.version metadata.language_info.pygments_lexer metadata.language_info.version metadata.toc metadata.notify_time metadata.varInspector cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell'
pre-commit install
