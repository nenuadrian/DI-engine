.PHONY: html clean serve

MKDOCS ?= mkdocs
BUILDDIR = _build
CONFIG_FILE = mkdocs.yml

html:
	$(MKDOCS) build --strict --config-file $(CONFIG_FILE) --site-dir $(BUILDDIR)/html

serve:
	$(MKDOCS) serve --config-file $(CONFIG_FILE)

clean:
	rm -rf $(BUILDDIR)
