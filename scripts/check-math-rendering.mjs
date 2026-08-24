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
  let displayDelimiterCount = 0
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

    if (trimmed === '\\[' || trimmed === '\\]') {
      unsupported.push({ line: index + 1, delimiter: trimmed })
    }

    displayDelimiterCount += line.match(/(?<!\\)\$\$/g)?.length ?? 0
  })

  return {
    unsupported,
    expectedDisplayMath: displayDelimiterCount / 2,
  }
}

const failures = []

for (const markdownFile of markdownFiles(notesRoot)) {
  const source = readFileSync(markdownFile, 'utf8')
  const sourcePath = relative(repositoryRoot, markdownFile)
  const { unsupported, expectedDisplayMath } = inspectSource(source)

  for (const issue of unsupported) {
    failures.push(
      `${sourcePath}:${issue.line} uses unsupported ${issue.delimiter} display-math delimiter`,
    )
  }

  if (!expectedDisplayMath) continue

  const outputFile = join(
    outputRoot,
    sourcePath.replace(/\.md$/, '.html'),
  )
  const output = readFileSync(outputFile, 'utf8')
  const renderedDisplayMath = output.match(/display="true"/g)?.length ?? 0

  if (renderedDisplayMath !== expectedDisplayMath) {
    failures.push(
      `${sourcePath} expected ${expectedDisplayMath} display formulas, rendered ${renderedDisplayMath}`,
    )
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log('Math rendering check passed.')
