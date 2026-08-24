import { readFileSync, readdirSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repositoryRoot = resolve(fileURLToPath(new URL('..', import.meta.url)))
const notesRoot = join(repositoryRoot, 'notes')
const outputRoot = join(repositoryRoot, '.vitepress', 'dist')

function markdownFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)

    if (entry.isDirectory()) return markdownFiles(path)
    if (entry.isFile() && entry.name.endsWith('.md')) return [path]
    return []
  })
}

function inspectSource(source) {
  const unsupported = []
  const unpaired = []
  let displayMathOpen = null
  let expectedDisplayMath = 0
  let expectedInlineMath = 0
  let fence = null

  source.split('\n').forEach((line, index) => {
    const trimmed = line.trim()
    const fenceMatch = trimmed.match(/^(`{3,}|~{3,})/)

    if (fenceMatch) {
      const marker = fenceMatch[1][0]
      fence = fence === marker ? null : marker
      return
    }

    if (fence) return

    const withoutInlineCode = line.replace(/(`+).*?\1/g, (value) =>
      ' '.repeat(value.length),
    )

    for (const match of withoutInlineCode.matchAll(
      /(?<!\\)\\(?:\[|\]|\(|\))/g,
    )) {
      unsupported.push({
        line: index + 1,
        column: (match.index ?? 0) + 1,
        delimiter: match[0],
      })
    }

    let inlineMathOpen = null

    for (let column = 0; column < withoutInlineCode.length; column += 1) {
      if (
        withoutInlineCode[column] === '\\' &&
        withoutInlineCode[column + 1] === '$'
      ) {
        column += 1
        continue
      }

      if (withoutInlineCode.slice(column, column + 2) === '$$') {
        if (inlineMathOpen) continue

        if (displayMathOpen) {
          expectedDisplayMath += 1
          displayMathOpen = null
        } else {
          displayMathOpen = { line: index + 1, column: column + 1 }
        }

        column += 1
        continue
      }

      if (displayMathOpen || withoutInlineCode[column] !== '$') continue

      const previous = withoutInlineCode[column - 1] ?? ''
      const next = withoutInlineCode[column + 1] ?? ''

      if (!inlineMathOpen) {
        if (next && !/\s/.test(next)) {
          inlineMathOpen = { line: index + 1, column: column + 1 }
        }
        continue
      }

      if (!/\s/.test(previous) && !/\d/.test(next)) {
        expectedInlineMath += 1
        inlineMathOpen = null
      }
    }

    if (inlineMathOpen) unpaired.push(inlineMathOpen)
  })

  return {
    unsupported,
    unpaired,
    displayMathOpen,
    expectedDisplayMath,
    expectedInlineMath,
  }
}

const failures = []
let totalDisplayMath = 0
let totalInlineMath = 0
const siteMarkdownFiles = [
  join(repositoryRoot, 'README.md'),
  join(repositoryRoot, 'index.md'),
  ...markdownFiles(notesRoot),
]

for (const markdownFile of siteMarkdownFiles) {
  const source = readFileSync(markdownFile, 'utf8')
  const sourcePath = relative(repositoryRoot, markdownFile)
  const {
    unsupported,
    unpaired,
    displayMathOpen,
    expectedDisplayMath,
    expectedInlineMath,
  } = inspectSource(source)
  totalDisplayMath += expectedDisplayMath
  totalInlineMath += expectedInlineMath

  for (const issue of unsupported) {
    failures.push(
      `${sourcePath}:${issue.line}:${issue.column} uses unsupported ${issue.delimiter} math delimiter`,
    )
  }

  for (const issue of unpaired) {
    failures.push(
      `${sourcePath}:${issue.line}:${issue.column} has an unpaired inline $ delimiter`,
    )
  }

  if (displayMathOpen) {
    failures.push(
      `${sourcePath}:${displayMathOpen.line}:${displayMathOpen.column} has an unpaired $$ delimiter`,
    )
  }

  const outputFile = join(
    outputRoot,
    sourcePath.replace(/\.md$/, '.html'),
  )
  const output = readFileSync(outputFile, 'utf8')
  const renderedDisplayMath = output.match(/display="true"/g)?.length ?? 0
  const renderedMath = output.match(/<mjx-container\b/g)?.length ?? 0
  const expectedMath = expectedDisplayMath + expectedInlineMath
  const visibleText = output
    .replace(
      /<(pre|code|script|style)\b[^>]*>[\s\S]*?<\/\1>/gi,
      '',
    )
    .replace(/<[^>]+>/g, ' ')
  const rawTexCommands = [
    ...visibleText.matchAll(/\\(?:[A-Za-z]+|\[|\]|\(|\))/g),
  ].map((match) => match[0])

  if (renderedDisplayMath !== expectedDisplayMath) {
    failures.push(
      `${sourcePath} expected ${expectedDisplayMath} display formulas, rendered ${renderedDisplayMath}`,
    )
  }

  if (renderedMath !== expectedMath) {
    failures.push(
      `${sourcePath} expected ${expectedMath} total formulas, rendered ${renderedMath}`,
    )
  }

  if (rawTexCommands.length) {
    failures.push(
      `${sourcePath} rendered raw TeX commands: ${[...new Set(rawTexCommands)].join(', ')}`,
    )
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log(
  `Math rendering check passed: ${siteMarkdownFiles.length} Markdown files, ${totalInlineMath} inline formulas, ${totalDisplayMath} display formulas.`,
)
