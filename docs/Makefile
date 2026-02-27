.PHONY: html html-strict clean serve

MKDOCS ?= mkdocs
BUILDDIR = _build
CONFIG_FILE = mkdocs.yml

html:
	$(MKDOCS) build --config-file $(CONFIG_FILE) --site-dir $(BUILDDIR)/html

html-strict:
	$(MKDOCS) build --strict --config-file $(CONFIG_FILE) --site-dir $(BUILDDIR)/html

serve:
	$(MKDOCS) serve --config-file $(CONFIG_FILE)

clean:
	rm -rf $(BUILDDIR)
