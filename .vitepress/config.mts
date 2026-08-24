import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { basename, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type DefaultTheme } from 'vitepress'

const repositoryRoot = resolve(fileURLToPath(new URL('..', import.meta.url)))

const sectionLabels: Record<string, string> = {
  agent: 'Agent 系统',
  compiler: '编译器',
  hypervisor: '虚拟化与系统安全',
  llm: '大语言模型',
  'llm-for-kernel': 'LLM × GPU Kernel',
}

const directoryLabels: Record<string, string> = {
  'deepseek-dsa': 'DeepSeek Sparse Attention',
  'deepseek-v4': 'DeepSeek V4',
  'event-tensor': 'Event Tensor',
  hida: 'HIDA',
  honeycomb: 'Honeycomb',
  'knowledge-distillation': '知识蒸馏',
  megamoe: 'MegaMoE',
  'minimax-msa': 'MiniMax Sparse Attention',
  nephele: 'Nephele',
  pithtrain: 'PithTrain',
  skvm: 'SkVM',
  'tiramisu-cgo': 'TIRAMISU',
  duvisor: 'DuVisor',
}

function cleanTitle(value: string) {
  return value
    .replace(/\s+#+\s*$/, '')
    .replace(/[`*_]/g, '')
    .trim()
}

function titleForMarkdown(file: string) {
  const source = readFileSync(file, 'utf8')
  const heading = source.match(/^#\s+(.+)$/m)?.[1]

  if (heading) return cleanTitle(heading)

  return basename(file, '.md')
    .replaceAll('-', ' ')
    .replaceAll('_', ' ')
}

function noteLink(file: string) {
  const pathFromRoot = relative(repositoryRoot, file)
    .split('\\')
    .join('/')
    .replace(/\.md$/, '')

  return `/${pathFromRoot}`
}

function pagePriority(fileName: string) {
  if (fileName === 'learning-path.md') return 0
  if (fileName === 'index.md') return 1
  return 10
}

function directoryItems(
  absoluteDirectory: string,
): DefaultTheme.SidebarItem[] {
  const entries = readdirSync(absoluteDirectory, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith('.'))

  const pages = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.md'))
    .sort((a, b) => {
      const priority = pagePriority(a.name) - pagePriority(b.name)
      return priority || a.name.localeCompare(b.name, 'zh-CN')
    })
    .map((entry) => {
      const file = join(absoluteDirectory, entry.name)
      return { text: titleForMarkdown(file), link: noteLink(file) }
    })

  const groups = entries
    .filter((entry) => entry.isDirectory())
    .sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
    .map((entry) => {
      const directory = join(absoluteDirectory, entry.name)
      const items = directoryItems(directory)

      if (!items.length) return undefined

      return {
        text: directoryLabels[entry.name] ?? entry.name.replaceAll('-', ' '),
        collapsed: true,
        items,
      } satisfies DefaultTheme.SidebarItem
    })
    .filter((item): item is DefaultTheme.SidebarItem => Boolean(item))

  return [...pages, ...groups]
}

function buildSidebar(): DefaultTheme.SidebarItem[] {
  const sectionOrder = ['llm', 'llm-for-kernel', 'agent', 'compiler', 'hypervisor']

  const sections = sectionOrder.map((section) => {
    const directory = join(repositoryRoot, 'notes', section)
    return {
      text: sectionLabels[section],
      collapsed: section !== 'llm',
      items: existsSync(directory) ? directoryItems(directory) : [],
    } satisfies DefaultTheme.SidebarItem
  })

  return [
    {
      text: '开始阅读',
      items: [
        { text: '知识地图', link: '/' },
        { text: '完整论文索引', link: '/README' },
        {
          text: 'LLM 推荐阅读路径',
          link: '/notes/llm/learning-path',
        },
      ],
    },
    ...sections,
  ]
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'Paper Reading',
  titleTemplate: ':title · Paper Reading',
  description: 'AI 系统、编译器、GPU Kernel 与虚拟化论文的结构化阅读笔记',
  base: '/Paper-reading/',
  lastUpdated: true,
  srcExclude: ['skills/**', 'tmp/**', 'node_modules/**'],
  ignoreDeadLinks: true,

  head: [
    ['meta', { name: 'theme-color', content: '#0f766e' }],
    ['meta', { name: 'author', content: 'KuangjuX' }],
  ],

  markdown: {
    math: true,
    lineNumbers: true,
    image: {
      lazyLoading: true,
    },
  },

  themeConfig: {
    siteTitle: 'Paper Reading',
    nav: [
      { text: '首页', link: '/' },
      { text: '论文索引', link: '/README' },
      { text: '学习路径', link: '/notes/llm/learning-path' },
      {
        text: '主题',
        items: [
          {
            text: 'LLM 系统',
            link: '/notes/llm/minimax-msa/msa',
          },
          {
            text: 'LLM × GPU Kernel',
            link: '/notes/llm-for-kernel/avo',
          },
          { text: 'Agent 系统', link: '/notes/agent/skvm/skvm' },
          { text: '编译器', link: '/notes/compiler/hida/hida' },
          {
            text: '虚拟化与安全',
            link: '/notes/hypervisor/duvisor/duvisor',
          },
        ],
      },
    ],

    sidebar: buildSidebar(),

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索笔记',
            buttonAriaLabel: '搜索笔记',
          },
          modal: {
            noResultsText: '没有找到相关内容',
            resetButtonTitle: '清除查询',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
            },
          },
        },
      },
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/KuangjuX/Paper-reading' },
    ],

    editLink: {
      pattern:
        'https://github.com/KuangjuX/Paper-reading/edit/main/:path',
      text: '在 GitHub 上编辑此页',
    },

    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'long',
      },
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部',
    langMenuLabel: '语言',

    footer: {
      message: '从论文出发，追到算法、系统与实现细节。',
      copyright: 'Copyright © KuangjuX',
    },
  },
})
