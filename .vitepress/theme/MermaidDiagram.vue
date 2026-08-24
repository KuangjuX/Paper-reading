<script lang="ts">
import {
  computed,
  defineComponent,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'
import { useData } from 'vitepress'

let nextDiagramId = 0
let renderQueue: Promise<void> = Promise.resolve()

export default defineComponent({
  name: 'MermaidDiagram',
  props: {
    code: {
      type: String,
      required: true,
    },
  },
  setup(props) {
    const { isDark } = useData()
    const canvas = ref<HTMLElement>()
    const status = ref<'loading' | 'ready' | 'error'>('loading')
    const errorMessage = ref('')
    const source = computed(() => decodeURIComponent(props.code))
    let revision = 0
    let destroyed = false

    async function renderDiagram() {
      const currentRevision = ++revision
      const useDarkTheme = isDark.value
      status.value = 'loading'
      errorMessage.value = ''
      await nextTick()

      const target = canvas.value
      if (!target || destroyed) return

      const task = renderQueue.then(async () => {
        if (destroyed || currentRevision !== revision) return

        const { default: mermaid } = await import('mermaid')
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: useDarkTheme ? 'dark' : 'neutral',
          fontFamily:
            'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif',
          htmlLabels: true,
          flowchart: {
            useMaxWidth: false,
            wrappingWidth: 280,
          },
        })

        const diagramId = `paper-reading-mermaid-${nextDiagramId++}`
        const { svg, bindFunctions } = await mermaid.render(
          diagramId,
          source.value,
        )

        if (destroyed || currentRevision !== revision) return

        target.innerHTML = svg
        bindFunctions?.(target)
        status.value = 'ready'
      })

      renderQueue = task.catch(() => undefined)

      try {
        await task
      } catch (error) {
        if (destroyed || currentRevision !== revision) return

        errorMessage.value =
          error instanceof Error ? error.message : String(error)
        status.value = 'error'
      }
    }

    onMounted(renderDiagram)
    watch(isDark, renderDiagram)
    onBeforeUnmount(() => {
      destroyed = true
      revision += 1
    })

    return {
      canvas,
      errorMessage,
      source,
      status,
    }
  },
})
</script>

<template>
  <figure class="mermaid-diagram">
    <div
      ref="canvas"
      class="mermaid-diagram__canvas"
      :aria-busy="status === 'loading'"
      aria-label="Mermaid 流程图"
    />
    <p v-if="status === 'loading'" class="mermaid-diagram__status">
      正在渲染流程图…
    </p>
    <div v-else-if="status === 'error'" class="mermaid-diagram__error">
      <strong>流程图渲染失败：</strong>{{ errorMessage }}
      <pre><code>{{ source }}</code></pre>
    </div>
  </figure>
</template>
