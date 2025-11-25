import pandas as pd
import os


def load_description(ref):
    desc_path = os.path.join('descriptions', f'{ref}.txt')
    if os.path.exists(desc_path):
        with open(desc_path, encoding='utf-8') as f:
            return f.read()
    return ""


def extract_keyword_features(text):
    text_lower = text.lower()

    luxury_keywords = ['luxury', 'exclusive', 'premium', 'high-end', 'prestigious',
                       'elegant', 'sophisticated', 'upscale', 'deluxe', 'exceptional']
    modern_keywords = ['modern', 'contemporary', 'renovated', 'new', 'updated',
                       'stylish', 'designer', 'state-of-the-art', 'sleek']
    view_keywords = ['sea view', 'mountain view', 'panoramic', 'views', 'vista',
                     'overlooking', 'scenic']
    outdoor_keywords = ['pool', 'garden', 'terrace', 'balcony', 'patio', 'outdoor',
                        'yard', 'deck']
    condition_keywords = ['excellent condition', 'well-maintained', 'pristine',
                          'immaculate', 'perfect', 'refurbished']
    location_keywords = ['beach', 'center', 'central', 'walking distance', 'close to',
                         'convenient', 'accessible', 'prime location']

    features = {
        'luxury_keyword_count': sum(1 for kw in luxury_keywords if kw in text_lower),
        'modern_keyword_count': sum(1 for kw in modern_keywords if kw in text_lower),
        'view_keyword_count': sum(1 for kw in view_keywords if kw in text_lower),
        'outdoor_keyword_count': sum(1 for kw in outdoor_keywords if kw in text_lower),
        'condition_keyword_count': sum(1 for kw in condition_keywords if kw in text_lower),
        'location_keyword_count': sum(1 for kw in location_keywords if kw in text_lower),
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': sum(len(w) for w in text.split()) / len(text.split()) if text.split() else 0,
        'exclamation_count': text.count('!'),
    }

    return features


def main():
    print("Loading properties...")
    df = pd.read_csv('properties.csv', names=[
        'reference','location','price','title','bedrooms','bathrooms',
        'indoor_area','outdoor_area','features'
    ])
    df['description'] = df['reference'].apply(load_description)
    print(f"Total properties: {len(df)}")

    print("\nCreating combined text...")
    df['full_text'] = (
        df['title'].astype(str) + " " +
        df['features'].astype(str) + " " +
        df['description'].astype(str)
    )

    print("\nExtracting keyword features...")
    keyword_features = df['full_text'].apply(extract_keyword_features)
    keyword_df = pd.DataFrame(keyword_features.tolist())

    print("\nCombining text features...")
    text_features_df = pd.concat([
        df[['reference']],
        keyword_df
    ], axis=1)

    text_features_df.to_csv('text_features.csv', index=False)

    print(f"\n[OK] Generated {len(keyword_df.columns)} keyword features")
    print("[OK] Saved to text_features.csv")


if __name__ == "__main__":
    main()
