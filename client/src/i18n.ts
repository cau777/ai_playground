import i18n from "i18next"
import LanguageDetector from "i18next-browser-languagedetector"
import { initReactI18next } from "react-i18next"
import HttpBackend from "i18next-http-backend";


i18n
    .use(LanguageDetector)
    .use(initReactI18next)
    .use(HttpBackend)
    .init({
        debug: import.meta.env.DEV,
        fallbackLng: "en",
        interpolation: {
            escapeValue: false,
        },
        backend: {
            loadPath: import.meta.env.BASE_URL + "locales/{{lng}}/{{ns}}.json"
        }
    })
    .then();

export default i18n;