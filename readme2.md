# Worker Registry & Video Consent Manager

A modern, fast, and secure web application built to streamline the registration of workers and explicitly track their consent for video recordings. Designed with researchers and administrators in mind, this tool ensures compliant, reliable, and user-friendly consent management.

## ✨ Key Features

- **📝 Worker Registration**: Easily onboard new workers into the system with their relevant details.
- **📹 Video Consent Tracking**: Digitally request, track, and manage video recording consent securely.
- **🛡️ Privacy-First & Compliant**: Ensures clear communication of consent terms and secure record-keeping.
- **⚡ Lightning Fast UI**: Built with React and Vite for near-instant load times and smooth interactions.

## 🛠️ Technology Stack

- **Frontend**: [React](https://react.dev/)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/) for robust, type-safe code.
- **Styling**: Vanilla CSS / Standard UI libraries.
- **Icons**: [Lucide React](https://lucide.dev/) for crisp, modern iconography.

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
- [Node.js](https://nodejs.org/en/) (v18 or higher recommended)
- `npm` or `yarn` package manager

### 1. Installation

Clone the repository and install dependencies:

```bash
# Navigate to the project directory
cd worker_registry

# Install dependencies
npm install
```

### 2. Environment Configuration

Create a `.env` file in the root directory if you are running this locally. An example production environment file (`.env.production`) is already provided.

```bash
# Example .env file
VITE_API_URL=http://localhost:8080/api  # Local development API
```

### 3. Start the Development Server

Start the Vite development server with Hot Module Replacement (HMR):

```bash
npm run dev
```

The application will be available at `http://localhost:5173`.

## 📦 Build for Production

To create an optimized production build:

```bash
npm run build
```

This will generate a `dist/` directory containing minified, optimized static assets ready to be deployed to your hosting provider (e.g., AWS, Vercel, Netlify).

To preview the production build locally:
```bash
npm run preview
```

## 📂 Project Structure

```
worker_registry/
├── src/
│   ├── App.tsx          # Main Application Component
│   ├── main.tsx         # React Entry Point
│   ├── index.css        # Global Styles
│   └── components/      # UI Components (Forms, Modals, etc.)
├── public/              # Static Assets
├── .env.production      # Production environment variables
├── package.json         # Project Dependencies & Scripts
├── tsconfig.json        # TypeScript Configuration
└── vite.config.ts       # Vite Configuration
```

## 🤝 Contributing

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.
