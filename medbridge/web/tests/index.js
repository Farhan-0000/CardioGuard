// Entry point so `node --test medbridge/web/tests` works on Node versions that
// treat a directory argument as a module path (it resolves to this index.js).
// Older Node versions scan the directory and run the *.test.mjs files directly.
import './markdown.test.mjs';
import './sse.test.mjs';
import './store.test.mjs';
import './helpers.test.mjs';
