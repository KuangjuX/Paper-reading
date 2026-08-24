import { readFileSync, readdirSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import DOMPurify from 'dompurify'

// Mermaid expects a browser-created DOMPurify instance even when only parsing.
// The CI check never renders HTML, so these no-op hooks keep the parser usable
// in Node without weakening the browser runtime's strict sanitization.
DOMPurify.addHook = () => undefined
DOMPurify.sanitize = (value) => value

const { default: mermaid } = await import('mermaid')

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

function diagramsInSource(source) {
  const diagrams = []
  const lines = source.split('\n')
  let fence = null

  lines.forEach((line, index) => {
    const opening = line.match(/^\s*(`{3,}|~{3,})\s*([^\s]*)\s*$/)

    if (!fence && opening) {
      fence = {
        character: opening[1][0],
        length: opening[1].length,
        language: opening[2],
        line: index + 1,
        source: [],
      }
      return
    }

    if (!fence) return

    const closing = line.match(/^\s*(`{3,}|~{3,})\s*$/)
    if (
      closing &&
      closing[1][0] === fence.character &&
      closing[1].length >= fence.length
    ) {
      if (fence.language === 'mermaid') {
        diagrams.push({
          line: fence.line,
          source: fence.source.join('\n').trim(),
        })
      }
      fence = null
      return
    }

    fence.source.push(line)
  })

  return diagrams
}

const failures = []
let totalDiagrams = 0
const siteMarkdownFiles = [
  join(repositoryRoot, 'README.md'),
  join(repositoryRoot, 'index.md'),
  ...markdownFiles(notesRoot),
]

for (const markdownFile of siteMarkdownFiles) {
  const sourcePath = relative(repositoryRoot, markdownFile)
  const source = readFileSync(markdownFile, 'utf8')
  const diagrams = diagramsInSource(source)
  totalDiagrams += diagrams.length

  for (const diagram of diagrams) {
    try {
      await mermaid.parse(diagram.source)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      failures.push(`${sourcePath}:${diagram.line} has invalid Mermaid syntax: ${message}`)
    }
  }

  const outputFile = join(outputRoot, sourcePath.replace(/\.md$/, '.html'))
  const output = readFileSync(outputFile, 'utf8')
  const renderedComponents =
    output.match(/class="mermaid-diagram"/g)?.length ?? 0
  const plainMermaidBlocks = output.match(/language-mermaid/g)?.length ?? 0

  if (renderedComponents !== diagrams.length) {
    failures.push(
      `${sourcePath} expected ${diagrams.length} Mermaid components, rendered ${renderedComponents}`,
    )
  }

  if (plainMermaidBlocks) {
    failures.push(
      `${sourcePath} left ${plainMermaidBlocks} Mermaid diagrams as plain code blocks`,
    )
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log(
  `Mermaid rendering check passed: ${siteMarkdownFiles.length} Markdown files, ${totalDiagrams} diagrams.`,
)
