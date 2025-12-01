/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/templates/**/*.html',
    './app/*/templates/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        // Sentiment colors
        'sentiment-positive': '#22c55e',
        'sentiment-negative': '#ef4444',
        'sentiment-neutral': '#64748b',
        // Emotion colors
        'emotion-joy': '#fbbf24',
        'emotion-anger': '#dc2626',
        'emotion-fear': '#7c3aed',
        'emotion-sadness': '#3b82f6',
        'emotion-surprise': '#f97316',
        'emotion-disgust': '#84cc16',
        'emotion-neutral': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 5px rgba(59, 130, 246, 0.5)' },
          '50%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.8)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
