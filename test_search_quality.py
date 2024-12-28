import os
import json
from datetime import datetime
from hybrid_search import TranscriptSearch
from collections import defaultdict
from typing import List, Dict, Any

class SearchQualityTester:
    def __init__(self):
        self.searcher = TranscriptSearch()
        self.test_results = defaultdict(list)
        self.load_test_data()

    def load_test_data(self):
        """Load the reference transcript data for testing"""
        json_path = "cache/https_www.telecomtv.com_content_digital-platforms-services_how-boost-mobile-is-managing-service-assurance-51783_.json"
        with open(json_path, 'r') as f:
            self.test_data = json.load(f)

    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable string with embedded metadata analysis"""
        formatted = []
        
        # First show embedded metadata if present
        if 'embedded_metadata' in metrics:
            formatted.append("\nEmbedded Metadata Analysis:")
            for category, values in metrics['embedded_metadata'].items():
                if values:
                    formatted.append(f"- {category.title()}: {', '.join(values)}")
            formatted.append("")  # Add spacing
        
        # Then show all numeric metrics
        formatted.append("Search Quality Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
            elif key != 'embedded_metadata':  # Skip embedded_metadata dict as it's handled above
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)

    def extract_speaker_from_text(self, text: str) -> str:
        """Extract speaker name from text based on common patterns"""
        # List of common speaker indicators
        indicators = ["said", "says", "asked", "mentioned", "explained", "noted", "added", "stated", "commented"]
        
        # Split text into sentences
        sentences = text.split('.')
        for sentence in sentences:
            # Look for names before common speaking verbs
            for indicator in indicators:
                if indicator in sentence.lower():
                    before_indicator = sentence.split(indicator)[0].strip()
                    # Take the last word or two as the potential name
                    words = before_indicator.split()
                    if len(words) >= 2:
                        return " ".join(words[-2:])
                    elif len(words) == 1:
                        return words[0]
        
        # If no clear speaker found, look for names at start of text
        words = text.strip().split()
        if len(words) >= 2:
            potential_name = " ".join(words[:2])
            if any(name in potential_name for name in ["Rahul", "Dawood", "Guy"]):  # Known speakers
                return potential_name
        
        return "Unknown Speaker"

    def format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results into a readable list"""
        output = [f"\nResults for query: '{query}'"]
        
        if not results:
            output.append("No results found")
            return "\n".join(output)
        
        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. Score: {result['similarity']:.3f}")
            output.append(f"   Text: {result['text']}")  # Show full text
            
            # Extract or use metadata
            speaker = (result.get('speaker') or 
                      self.extract_speaker_from_text(result['text']))
            company = result.get('company', "TelecomTV")  # Default company
            
            output.append(f"   Speaker: {speaker}")
            output.append(f"   Company: {company}")
        
        return "\n".join(output)

    def evaluate_results(self, results: List[Dict[str, Any]], query: str,
                        expected_content: str = None, expected_speaker: str = None, 
                        expected_company: str = None, expected_subjects: List[str] = None,
                        topic: str = None) -> Dict[str, float]:
        """
        Evaluate search results quality with enhanced verification of metadata matching and intent.
        
        Args:
            results: List of search results
            query: Original search query text
            expected_content: Expected content to find in results
            expected_speaker: Expected speaker name
            expected_company: Expected company name
            expected_subjects: Expected subjects/topics
            topic: General topic for relevance scoring
        """
        # Analyze query for embedded metadata
        query_lower = query.lower()
        embedded_metadata = {
            'speakers': [],
            'companies': [],
            'subjects': []
        }
        
        # Extract embedded metadata from query
        for speaker in self.test_data['transcript']:
            if speaker['metadata'].get('speaker') and speaker['metadata']['speaker'].lower() in query_lower:
                embedded_metadata['speakers'].append(speaker['metadata']['speaker'])
        
        for company in set(s['metadata'].get('company', '') for s in self.test_data['transcript']):
            if company and company.lower() in query_lower:
                embedded_metadata['companies'].append(company)
                
        for segment in self.test_data['transcript']:
            if segment['metadata'].get('subjects'):
                for subject in segment['metadata']['subjects']:
                    if subject.lower() in query_lower:
                        embedded_metadata['subjects'].append(subject)
        
        metrics = {
            'avg_score': sum(r['similarity'] for r in results) / len(results) if results else 0,
            'top_score': results[0]['similarity'] if results else 0,
            'result_count': len(results),
            'speaker_accuracy': 0.0,
            'company_accuracy': 0.0,
            'subject_accuracy': 0.0,
            'topic_relevance': 0.0,
            'embedded_metadata': embedded_metadata,
            'metadata_match_rate': 0.0
        }
        
        if results:
            # Calculate metadata match rates
            total_metadata_checks = 0
            total_metadata_matches = 0
            
            # Check embedded speakers
            if embedded_metadata['speakers']:
                total_metadata_checks += len(results)
                correct_speakers = sum(1 for r in results 
                                    if any(s.lower() == r.get('speaker', '').lower() 
                                          for s in embedded_metadata['speakers']))
                metrics['speaker_accuracy'] = correct_speakers / len(results)
                total_metadata_matches += correct_speakers
            
            # Check embedded companies
            if embedded_metadata['companies']:
                total_metadata_checks += len(results)
                correct_companies = sum(1 for r in results 
                                     if any(c.lower() == r.get('company', '').lower() 
                                           for c in embedded_metadata['companies']))
                metrics['company_accuracy'] = correct_companies / len(results)
                total_metadata_matches += correct_companies
            
            # Check embedded subjects
            if embedded_metadata['subjects']:
                total_metadata_checks += len(results)
                correct_subjects = sum(1 for r in results 
                                    if r.get('subjects') and 
                                    any(s.lower() in [sub.lower() for sub in r['subjects']]
                                        for s in embedded_metadata['subjects']))
                metrics['subject_accuracy'] = correct_subjects / len(results)
                total_metadata_matches += correct_subjects
            
            # Calculate overall metadata match rate
            if total_metadata_checks > 0:
                metrics['metadata_match_rate'] = total_metadata_matches / total_metadata_checks
            
            # Additional checks for explicitly expected values
            if expected_content:
                found = any(expected_content.lower() in r['text'].lower() for r in results)
                metrics['found_expected'] = 1.0 if found else 0.0
            
            if expected_speaker:
                correct_speaker = sum(1 for r in results 
                                   if r.get('speaker', '').lower() == expected_speaker.lower())
                metrics['explicit_speaker_accuracy'] = correct_speaker / len(results)
            
            if expected_company:
                correct_company = sum(1 for r in results 
                                   if r.get('company', '').lower() == expected_company.lower())
                metrics['explicit_company_accuracy'] = correct_company / len(results)
            
            if expected_subjects:
                correct_subjects = sum(1 for r in results 
                                    if r.get('subjects') and 
                                    any(s.lower() in [sub.lower() for sub in r['subjects']]
                                        for s in expected_subjects))
                metrics['explicit_subject_accuracy'] = correct_subjects / len(results)
            
            # Topic relevance using similarity score
            if topic:
                relevant_results = sum(1 for r in results if r['similarity'] > 0.5)
                metrics['topic_relevance'] = relevant_results / len(results)
        
        return metrics

    def run_company_tests(self):
        """Test searching for different companies mentioned"""
        print("\n" + "="*80)
        print("Testing Company Searches...")
        print("="*80)
        
        # Extract unique companies from test data
        companies = {segment['metadata']['company'] for segment in self.test_data['transcript'] 
                    if segment['metadata'].get('company')}
        
        for company in companies:
            # Test exact company name
            results = self.searcher.hybrid_search(
                search_text=f"statements from {company}",
                filters={'companies': [company]}
            )
            query = f"statements from {company}"
            metrics = self.evaluate_results(results, query, expected_company=company)
            self.test_results['company_searches'].append({
                'query': company,
                'metrics': metrics
            })
            formatted_results = self.format_search_results(results, f"statements from {company}")
            print(formatted_results)
            print("\nMetrics:")
            print(self.format_metrics(metrics))
            print("\n" + "-"*80 + "\n")

    def run_speaker_tests(self):
        """Test searching for different speakers"""
        print("\n" + "="*80)
        print("Testing Speaker Searches...")
        print("="*80)
        
        # Extract unique speakers
        speakers = {segment['metadata']['speaker'] for segment in self.test_data['transcript']
                   if segment['metadata'].get('speaker')}
        
        for speaker in speakers:
            # Test exact speaker name
            results = self.searcher.hybrid_search(
                search_text=f"statements by {speaker}",
                filters={'speakers': [speaker]}
            )
            query = f"role or position of {speaker}"
            metrics = self.evaluate_results(results, query, expected_speaker=speaker)
            self.test_results['speaker_searches'].append({
                'query': speaker,
                'metrics': metrics
            })
            formatted_results = self.format_search_results(results, f"statements by {speaker}")
            print(formatted_results)
            print("\nMetrics:")
            print(self.format_metrics(metrics))
            print("\n" + "-"*80 + "\n")

            # Test speaker's role/position if available
            results = self.searcher.hybrid_search(
                search_text=f"role or position of {speaker}",
                filters={'speakers': [speaker]}
            )
            query = f"role of {speaker}"
            self.test_results['speaker_role_searches'].append({
                'query': query,
                'metrics': self.evaluate_results(results, query)
            })

    def run_topic_tests(self):
        """Test searching for different topics and themes"""
        print("\n" + "="*80)
        print("Testing Topic Searches...")
        print("="*80)
        
        # Common telecom/tech topics to test
        topics = [
            "service assurance",
            "network performance",
            "customer experience",
            "mobile services",
            "technical challenges",
            "business strategy",
            "future plans",
            "market competition",
            "technology implementation",
            "operational efficiency"
        ]
        
        for topic in topics:
            results = self.searcher.hybrid_search(search_text=topic)
            query = topic
            metrics = self.evaluate_results(results, query, topic=topic)
            self.test_results['topic_searches'].append({
                'query': topic,
                'metrics': metrics
            })
            formatted_results = self.format_search_results(results, topic)
            print(formatted_results)
            print("\nMetrics:")
            print(self.format_metrics(metrics))
            print("\n" + "-"*80 + "\n")

    def run_emotion_sentiment_tests(self):
        """Test searching for emotional content and sentiment"""
        print("\n" + "="*80)
        print("Testing Emotion/Sentiment Searches...")
        print("="*80)
        
        sentiment_queries = [
            "positive statements about success",
            "challenges or difficulties faced",
            "optimistic views about future",
            "concerns or worries expressed",
            "confident statements",
            "cautious or careful statements"
        ]
        
        for query in sentiment_queries:
            results = self.searcher.hybrid_search(search_text=query)
            metrics = self.evaluate_results(results, query)
            self.test_results['sentiment_searches'].append({
                'query': query,
                'metrics': metrics
            })
            formatted_results = self.format_search_results(results, query)
            print(formatted_results)
            print("\nMetrics:")
            print(self.format_metrics(metrics))
            print("\n" + "-"*80 + "\n")

    def run_metadata_tests(self):
        """Test searching based on metadata characteristics"""
        print("\n" + "="*80)
        print("Testing Metadata-based Searches...")
        print("="*80)
        
        # Test duration-based queries
        duration_queries = [
            ("longer statements lasting more than 30 seconds", {'min_duration': 30}),
            ("brief quick statements", {'max_duration': 15}),
            ("detailed explanations", {'min_duration': 45}),
            ("short answers", {'max_duration': 10})
        ]
        
        for query, duration_filter in duration_queries:
            results = self.searcher.hybrid_search(
                search_text=query,
                filters=duration_filter
            )
            metrics = self.evaluate_results(results, query)
            self.test_results['duration_searches'].append({
                'query': query,
                'metrics': metrics
            })
            formatted_results = self.format_search_results(results, query)
            print(formatted_results)
            print("\nMetrics:")
            print(self.format_metrics(metrics))
            print("\n" + "-"*80 + "\n")

    def run_speaker_company_combination_tests(self):
        """Test various combinations of speakers and companies with different topics"""
        print("\n" + "="*80)
        print("Testing Speaker-Company Combinations...")
        print("="*80)
        
        # Extract unique speakers and companies
        speakers = {segment['metadata']['speaker'] for segment in self.test_data['transcript']
                   if segment['metadata'].get('speaker')}
        companies = {segment['metadata']['company'] for segment in self.test_data['transcript'] 
                    if segment['metadata'].get('company')}
        
        # Topics to test with
        topics = [
            "network performance",
            "customer experience",
            "technical challenges",
            "future plans",
            "market strategy"
        ]
        
        # Test single speaker, single company
        for speaker in speakers:
            for company in companies:
                for topic in topics:
                    query = f"{topic} discussed by {speaker} from {company}"
                    results = self.searcher.hybrid_search(
                        search_text=query,
                        filters={
                            'speakers': [speaker],
                            'companies': [company]
                        }
                    )
                    metrics = self.evaluate_results(
                        results,
                        query,
                        expected_speaker=speaker,
                        expected_company=company,
                        topic=topic
                    )
                    self.test_results['speaker_company_single'].append({
                        'query': query,
                        'speaker': speaker,
                        'company': company,
                        'topic': topic,
                        'metrics': metrics
                    })
                    formatted_results = self.format_search_results(results, query)
                    print(formatted_results)
                    print("\nMetrics:")
                    print(self.format_metrics(metrics))
                    print("\n" + "-"*80 + "\n")
        
        # Test multiple speakers, single company
        speaker_list = list(speakers)
        for i in range(0, len(speaker_list), 2):
            if i + 1 < len(speaker_list):
                speakers_pair = [speaker_list[i], speaker_list[i+1]]
                for company in companies:
                    for topic in topics:
                        query = f"{topic} discussed by {speakers_pair[0]} or {speakers_pair[1]} from {company}"
                        results = self.searcher.hybrid_search(
                            search_text=query,
                            filters={'companies': [company]}
                        )
                        metrics = self.evaluate_results(
                            results,
                            query,
                            expected_company=company,
                            topic=topic
                        )
                        self.test_results['speaker_company_multi_speaker'].append({
                            'query': query,
                            'speakers': speakers_pair,
                            'company': company,
                            'topic': topic,
                            'metrics': metrics
                        })
                        formatted_results = self.format_search_results(results, query)
                        print(formatted_results)
                        print("\nMetrics:")
                        print(self.format_metrics(metrics))
                        print("\n" + "-"*80 + "\n")
        
        # Test single speaker, multiple companies
        company_list = list(companies)
        for speaker in speakers:
            for i in range(0, len(company_list), 2):
                if i + 1 < len(company_list):
                    companies_pair = [company_list[i], company_list[i+1]]
                    for topic in topics:
                        query = f"{topic} discussed by {speaker} from {companies_pair[0]} or {companies_pair[1]}"
                        results = self.searcher.hybrid_search(
                            search_text=query,
                            filters={'speakers': [speaker]}
                        )
                        metrics = self.evaluate_results(
                            results,
                            query,
                            expected_speaker=speaker,
                            topic=topic
                        )
                        self.test_results['speaker_company_multi_company'].append({
                            'query': query,
                            'speaker': speaker,
                            'companies': companies_pair,
                            'topic': topic,
                            'metrics': metrics
                        })
                        formatted_results = self.format_search_results(results, query)
                        print(formatted_results)
                        print("\nMetrics:")
                        print(self.format_metrics(metrics))
                        print("\n" + "-"*80 + "\n")

    def run_complex_queries(self):
        """Test complex multi-aspect queries with enhanced verification"""
        print("\n" + "="*80)
        print("Testing Complex Queries...")
        print("="*80)
        
        complex_queries = [
            {
                'query': "positive statements about customer experience from senior executives at Boost Mobile",
                'company': "Boost Mobile",
                'topic': "customer experience"
            },
            {
                'query': "technical challenges in service assurance discussed by Boost Mobile leadership",
                'company': "Boost Mobile",
                'topic': "technical challenges"
            },
            {
                'query': "future plans for network improvement mentioned by CTOs",
                'topic': "network improvement"
            },
            {
                'query': "competitive advantages described by company leadership in telecommunications",
                'topic': "competitive advantages"
            },
            {
                'query': "specific examples of successful implementations from technical directors",
                'topic': "successful implementations"
            }
        ]
        
        for query_info in complex_queries:
            filters = {}
            if 'company' in query_info:
                filters['companies'] = [query_info['company']]
                
            results = self.searcher.hybrid_search(
                search_text=query_info['query'],
                filters=filters
            )
            metrics = self.evaluate_results(
                results,
                query_info['query'],
                expected_company=query_info.get('company'),
                topic=query_info['topic']
            )
            self.test_results['complex_searches'].append({
                'query': query_info['query'],
                'metrics': metrics
            })
            formatted_results = self.format_search_results(results, query_info['query'])
            print(formatted_results)
            print("\nMetrics:")
            print(self.format_metrics(metrics))
            print("\n" + "-"*80 + "\n")

    def calculate_aggregate_metrics(self):
        """Calculate overall metrics across all test categories"""
        print("\n" + "="*80)
        print("Calculating Aggregate Metrics...")
        print("="*80)
        
        aggregates = {}
        for category, results in self.test_results.items():
            category_metrics = {
                'avg_score': sum(r['metrics']['avg_score'] for r in results) / len(results),
                'avg_top_score': sum(r['metrics']['top_score'] for r in results) / len(results),
                'avg_result_count': sum(r['metrics']['result_count'] for r in results) / len(results),
                'total_queries': len(results)
            }
            aggregates[category] = category_metrics
            
            print(f"\n{category.replace('_', ' ').title()} Aggregate Metrics:")
            print(self.format_metrics(category_metrics))
            print("\n" + "-"*80)
        
        return aggregates

    def run_all_tests(self):
        """Run all test categories and return comprehensive results"""
        print("\n" + "="*80)
        print("Starting comprehensive search quality testing...")
        print("="*80 + "\n")
        
        # Run all test categories
        self.run_company_tests()
        self.run_speaker_tests()
        self.run_topic_tests()
        self.run_emotion_sentiment_tests()
        self.run_metadata_tests()
        self.run_speaker_company_combination_tests()
        self.run_complex_queries()
        
        # Calculate and return aggregate metrics
        return self.calculate_aggregate_metrics()

def execute_single_query(query: str):
    """Execute a single search query and display results"""
    tester = SearchQualityTester()
    results = tester.searcher.hybrid_search(search_text=query)
    print(tester.format_search_results(results, query))
    metrics = tester.evaluate_results(results, query)
    print("\nMetrics:")
    print(tester.format_metrics(metrics))

def main():
    import sys
    
    if len(sys.argv) > 1:
        # If argument provided, treat it as a search query
        query = " ".join(sys.argv[1:])  # Join all arguments as they might contain spaces
        execute_single_query(query)
    else:
        # Run full test suite
        tester = SearchQualityTester()
        aggregate_metrics = tester.run_all_tests()
        
        print("\n" + "="*80)
        print("Test Suite Execution Complete")
        print("="*80)
        print("\nOverall Search System Quality Assessment:")
        
        # Calculate global averages
        all_scores = []
        all_top_scores = []
        all_result_counts = []
        
        for category_metrics in aggregate_metrics.values():
            all_scores.append(category_metrics['avg_score'])
            all_top_scores.append(category_metrics['avg_top_score'])
            all_result_counts.append(category_metrics['avg_result_count'])
        
        print("\nGlobal Metrics:")
        print(f"Average Score Across All Categories: {sum(all_scores) / len(all_scores):.3f}")
        print(f"Average Top Score Across All Categories: {sum(all_top_scores) / len(all_top_scores):.3f}")
        print(f"Average Result Count Across All Categories: {sum(all_result_counts) / len(all_result_counts):.3f}")
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
