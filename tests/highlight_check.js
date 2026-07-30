/* Extract highlightSql from index.html and check it on real generated SQL. */
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];

// Pull out just the three functions we need.
const grab = (name) => {
  const i = src.indexOf("function " + name);
  let depth = 0, j = src.indexOf("{", i);
  for (let k = j; k < src.length; k++) {
    if (src[k] === "{") depth++;
    else if (src[k] === "}") { depth--; if (!depth) return src.slice(i, k + 1); }
  }
};
const kw = src.match(/var SQL_KEYWORDS = [^\n]+/)[0];
const fn = src.match(/var SQL_FUNCS = [^\n]+/)[0];
eval(kw + "\n" + fn + "\n" + grab("escapeHtml") + "\n" + grab("highlightSql"));

let fails = 0;
function check(name, cond, extra) {
  console.log((cond ? "PASS " : "FAIL ") + name + (extra ? "  " + extra : ""));
  if (!cond) fails++;
}

const CASES = [
  "SELECT COUNT(*) FROM students WHERE semester = 6",
  "SELECT sub.name AS subject, m.exam_type, m.score, m.max_score FROM marks m JOIN subjects sub ON sub.id = m.subject_id",
  "SELECT cgpa FROM cgpa",
  "SELECT s.name, AVG(100.0 * m.score / m.max_score) AS avg_percent FROM students s INNER JOIN marks m ON m.student_id = s.id WHERE s.name = 'Artificial Intelligence' GROUP BY s.id ORDER BY avg_percent DESC LIMIT 3",
];

CASES.forEach((sql, i) => {
  const out = highlightSql(sql);
  check(`case ${i}: no leftover placeholder`, !/@@STR/.test(out), out.slice(0, 60));
  check(`case ${i}: tags are balanced`,
        (out.match(/<span/g) || []).length === (out.match(/<\/span>/g) || []).length);
  check(`case ${i}: no raw < or > injected`, !/<(?!\/?span)/.test(out));
  // Strip the markup and it must equal the escaped original.
  const plain = out.replace(/<\/?span[^>]*>/g, "");
  check(`case ${i}: text survives round-trip`, plain === escapeHtml(sql),
        plain === escapeHtml(sql) ? "" : "\n   got: " + plain + "\n   want:" + escapeHtml(sql));
});

// Keywords inside a string literal must not be recoloured.
const tricky = highlightSql("SELECT name FROM subjects WHERE name = 'ORDER BY 5'");
check("keywords inside a literal stay plain",
      /<span class="str">&#39;ORDER BY 5&#39;<\/span>/.test(tricky), tricky);

// XSS: an injected tag must be escaped, never live markup.
const evil = highlightSql("SELECT '<img src=x onerror=alert(1)>' FROM t");
check("html in the SQL is escaped", !/<img/.test(evil), evil.slice(0, 90));

console.log(fails ? `\n${fails} FAILURES` : "\nALL HIGHLIGHT CHECKS PASSED");
process.exit(fails ? 1 : 0);
