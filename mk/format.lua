-- Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
-- SPDX-License-Identifier: Apache-2.0
--
-- Project indent style, loaded by `make format` (nvim --headless --clean).
-- Mirrors the maintainer's editor setup, so formatting on save and
-- `make format` produce identical files:
--   * Vim's C indenter (gg=G), one tab per level (noexpandtab, ts = sw = 8)
--   * N-s: no indent inside namespaces
--   * E-s: no indent inside extern "C" blocks
--   * g0 : public:/private:/protected: aligned with the class keyword
--   * the line after a `template <...>` line keeps the template's indent
function _G.CppIndent()
	local prev = vim.fn.prevnonblank(vim.v.lnum - 1)
	if prev > 0 and vim.fn.getline(prev):match('^%s*template%s*<.*>%s*$') then
		return vim.fn.indent(prev)
	end
	return vim.fn.cindent(vim.v.lnum)
end

vim.cmd('filetype plugin indent on')
vim.api.nvim_create_autocmd('FileType', {
	pattern = { 'c', 'cpp' },
	callback = function()
		vim.opt_local.cinoptions:append('N-s,E-s,g0')
		vim.opt_local.indentexpr = 'v:lua.CppIndent()'
		vim.opt_local.expandtab = false
		vim.opt_local.tabstop = 8
		vim.opt_local.shiftwidth = 8
	end,
})
