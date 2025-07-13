"""
Internationalization (i18n) support for multi-language functionality.
Supports Arabic, English, and French languages.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    FRENCH = "fr"
    ARABIC = "ar"


class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_language: Language = Language.ENGLISH):
        self.current_language = default_language
        self.translations = self._load_translations()
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load all translations."""
        return {
            # UI Labels and Messages
            "app_title": {
                "en": "CV Evaluation System",
                "fr": "Système d'Évaluation de CV",
                "ar": "نظام تقييم السيرة الذاتية"
            },
            "document_chat_title": {
                "en": "AI Document Chat Assistant",
                "fr": "Assistant de Chat IA pour Documents",
                "ar": "مساعد الدردشة الذكي للمستندات"
            },
            "upload_document": {
                "en": "Upload Document",
                "fr": "Télécharger un Document",
                "ar": "رفع مستند"
            },
            "supported_formats": {
                "en": "Supported File Types",
                "fr": "Types de Fichiers Supportés",
                "ar": "أنواع الملفات المدعومة"
            },
            "document_summary": {
                "en": "Document Summary",
                "fr": "Résumé du Document",
                "ar": "ملخص المستند"
            },
            "chat_about_document": {
                "en": "Chat about this Document",
                "fr": "Discuter de ce Document",
                "ar": "تحدث عن هذا المستند"
            },
            "quick_questions": {
                "en": "Quick Questions",
                "fr": "Questions Rapides",
                "ar": "أسئلة سريعة"
            },
            "ask_question": {
                "en": "Ask a question about this document:",
                "fr": "Posez une question sur ce document:",
                "ar": "اطرح سؤالاً حول هذا المستند:"
            },
            "send": {
                "en": "Send",
                "fr": "Envoyer",
                "ar": "إرسال"
            },
            "clear_chat": {
                "en": "Clear Chat",
                "fr": "Effacer la Discussion",
                "ar": "مسح المحادثة"
            },
            "processing": {
                "en": "Processing document...",
                "fr": "Traitement du document...",
                "ar": "معالجة المستند..."
            },
            "success_processed": {
                "en": "processed successfully",
                "fr": "traité avec succès",
                "ar": "تمت المعالجة بنجاح"
            },
            "error_processing": {
                "en": "Failed to process document",
                "fr": "Échec du traitement du document",
                "ar": "فشل في معالجة المستند"
            },
            "upload_prompt": {
                "en": "Please upload a document to start chatting about its content",
                "fr": "Veuillez télécharger un document pour commencer à discuter de son contenu",
                "ar": "يرجى رفع مستند لبدء الحديث عن محتواه"
            },
            
            # Quick Questions for different document types
            "cv_overall_assessment": {
                "en": "What is your overall assessment of this candidate?",
                "fr": "Quelle est votre évaluation globale de ce candidat?",
                "ar": "ما هو تقييمك الشامل لهذا المرشح؟"
            },
            "cv_key_skills": {
                "en": "What are the key skills mentioned in this document?",
                "fr": "Quelles sont les compétences clés mentionnées dans ce document?",
                "ar": "ما هي المهارات الأساسية المذكورة في هذا المستند؟"
            },
            "cv_experience_level": {
                "en": "What is the experience level of this candidate?",
                "fr": "Quel est le niveau d'expérience de ce candidat?",
                "ar": "ما هو مستوى خبرة هذا المرشح؟"
            },
            "excel_data_summary": {
                "en": "Can you summarize the data in this spreadsheet?",
                "fr": "Pouvez-vous résumer les données de cette feuille de calcul?",
                "ar": "هل يمكنك تلخيص البيانات في هذا الجدول؟"
            },
            "excel_key_metrics": {
                "en": "What are the key metrics or numbers in this data?",
                "fr": "Quelles sont les métriques ou chiffres clés dans ces données?",
                "ar": "ما هي المقاييس أو الأرقام الرئيسية في هذه البيانات؟"
            },
            "excel_trends": {
                "en": "What trends or patterns can you identify in this data?",
                "fr": "Quelles tendances ou modèles pouvez-vous identifier dans ces données?",
                "ar": "ما هي الاتجاهات أو الأنماط التي يمكنك تحديدها في هذه البيانات؟"
            },
            "word_main_points": {
                "en": "What are the main points discussed in this document?",
                "fr": "Quels sont les points principaux discutés dans ce document?",
                "ar": "ما هي النقاط الرئيسية المناقشة في هذا المستند؟"
            },
            "word_purpose": {
                "en": "What is the purpose or objective of this document?",
                "fr": "Quel est le but ou l'objectif de ce document?",
                "ar": "ما هو الغرض أو الهدف من هذا المستند؟"
            },
            "word_summary": {
                "en": "Can you provide a summary of this document?",
                "fr": "Pouvez-vous fournir un résumé de ce document?",
                "ar": "هل يمكنك تقديم ملخص لهذا المستند؟"
            },
            "generic_summary": {
                "en": "Can you summarize this document?",
                "fr": "Pouvez-vous résumer ce document?",
                "ar": "هل يمكنك تلخيص هذا المستند؟"
            },
            "generic_key_points": {
                "en": "What are the key points in this document?",
                "fr": "Quels sont les points clés de ce document?",
                "ar": "ما هي النقاط الرئيسية في هذا المستند؟"
            },
            "generic_analysis": {
                "en": "Can you analyze the content of this document?",
                "fr": "Pouvez-vous analyser le contenu de ce document?",
                "ar": "هل يمكنك تحليل محتوى هذا المستند؟"
            },
            
            # File type descriptions
            "pdf_document": {
                "en": "PDF Document",
                "fr": "Document PDF",
                "ar": "مستند PDF"
            },
            "word_document": {
                "en": "Word Document",
                "fr": "Document Word",
                "ar": "مستند Word"
            },
            "excel_spreadsheet": {
                "en": "Excel Spreadsheet",
                "fr": "Feuille de Calcul Excel",
                "ar": "جدول Excel"
            },
            "text_file": {
                "en": "Text File",
                "fr": "Fichier Texte",
                "ar": "ملف نصي"
            },
            
            # AI Model related
            "ai_model_settings": {
                "en": "AI Model Settings",
                "fr": "Paramètres du Modèle IA",
                "ar": "إعدادات نموذج الذكاء الاصطناعي"
            },
            "select_language": {
                "en": "Select Language",
                "fr": "Sélectionner la Langue",
                "ar": "اختر اللغة"
            },
            "language_english": {
                "en": "English",
                "fr": "Anglais",
                "ar": "الإنجليزية"
            },
            "language_french": {
                "en": "French",
                "fr": "Français",
                "ar": "الفرنسية"
            },
            "language_arabic": {
                "en": "Arabic",
                "fr": "Arabe",
                "ar": "العربية"
            }
        }
    
    def get_text(self, key: str, language: Optional[Language] = None) -> str:
        """Get translated text for a key."""
        lang = language or self.current_language
        lang_code = lang.value
        
        if key in self.translations:
            return self.translations[key].get(lang_code, self.translations[key].get("en", key))
        
        logger.warning(f"Translation key '{key}' not found")
        return key
    
    def set_language(self, language: Language):
        """Set the current language."""
        self.current_language = language
        logger.info(f"Language changed to: {language.value}")
    
    def get_current_language(self) -> Language:
        """Get the current language."""
        return self.current_language
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages with their display names."""
        return {
            Language.ENGLISH.value: self.get_text("language_english"),
            Language.FRENCH.value: self.get_text("language_french"),
            Language.ARABIC.value: self.get_text("language_arabic")
        }
    
    def get_ai_prompt_language_instruction(self, language: Optional[Language] = None) -> str:
        """Get language instruction for AI prompts."""
        lang = language or self.current_language
        
        instructions = {
            Language.ENGLISH: "Please respond in English.",
            Language.FRENCH: "Veuillez répondre en français.",
            Language.ARABIC: "يرجى الرد باللغة العربية."
        }
        
        return instructions.get(lang, instructions[Language.ENGLISH])


# Global i18n manager instance
i18n = I18nManager()
