<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit(0);
}

// ─── Naive Bayes Engine ───────────────────────────────────────────────────────

class NaiveBayesClassifier {
    private $classCounts = [];
    private $featureCounts = [];
    private $totalSamples = 0;
    private $features = [];

    public function train(array $data, array $features): void {
        $this->features = $features;
        foreach ($data as $row) {
            $label = $row['label'];
            $this->classCounts[$label] = ($this->classCounts[$label] ?? 0) + 1;
            $this->totalSamples++;
            foreach ($features as $feat) {
                $val = $row[$feat] ?? 'none';
                $this->featureCounts[$feat][$val][$label] =
                    ($this->featureCounts[$feat][$val][$label] ?? 0) + 1;
            }
        }
    }

    public function predict(array $input): array {
        $scores = [];
        foreach ($this->classCounts as $label => $count) {
            $prior = $count / $this->totalSamples;
            $logProb = log($prior);
            foreach ($this->features as $feat) {
                $val = $input[$feat] ?? 'none';
                $featureCount = $this->featureCounts[$feat][$val][$label] ?? 0;
                // Laplace smoothing
                $vocabSize = count($this->featureCounts[$feat] ?? []) + 1;
                $prob = ($featureCount + 1) / ($count + $vocabSize);
                $logProb += log($prob);
            }
            $scores[$label] = $logProb;
        }

        // Convert log probs to probabilities
        $maxLog = max($scores);
        $expScores = [];
        $sumExp = 0;
        foreach ($scores as $label => $logProb) {
            $expScores[$label] = exp($logProb - $maxLog);
            $sumExp += $expScores[$label];
        }

        $probabilities = [];
        foreach ($expScores as $label => $expScore) {
            $probabilities[$label] = round(($expScore / $sumExp) * 100, 2);
        }

        arsort($probabilities);
        return $probabilities;
    }
}

// ─── Load & Parse CSV ─────────────────────────────────────────────────────────

function loadCSV(string $path): array {
    if (!file_exists($path)) return [];
    $rows = [];
    $handle = fopen($path, 'r');
    $headers = fgetcsv($handle);
    // Trim BOM and whitespace
    $headers = array_map(fn($h) => trim($h, "\xEF\xBB\xBF\r\n "), $headers);
    while (($line = fgetcsv($handle)) !== false) {
        if (count($line) !== count($headers)) continue;
        $row = array_combine($headers, $line);
        $rows[] = $row;
    }
    fclose($handle);
    return [$headers, $rows];
}

// ─── Route Handlers ───────────────────────────────────────────────────────────

$action = $_GET['action'] ?? 'predict';

if ($action === 'status') {
    $cacheFile = __DIR__ . '/model_cache.json';
    $trained   = file_exists($cacheFile);
    $info = [];
    if ($trained) {
        $model = json_decode(file_get_contents($cacheFile), true);
        $info  = ['sample_count' => $model['totalSamples'] ?? 0, 'trained_at' => $model['trained_at'] ?? null];
    }
    echo json_encode(['trained' => $trained, 'csv_exists' => file_exists(__DIR__ . '/disease_data.csv'), 'info' => $info]);
    exit;
}

if ($action === 'train') {
    $csvPath = __DIR__ . '/disease_data.csv';
    [$headers, $data] = loadCSV($csvPath);
    if (empty($data)) {
        echo json_encode(['error' => 'CSV not found or empty', 'path' => $csvPath]);
        exit;
    }
    $features = array_filter($headers, fn($h) => $h !== 'label' && $h !== 'duration');
    $features = array_values($features);

    $nb = new NaiveBayesClassifier();
    $nb->train($data, $features);

    // Cache model
    file_put_contents(__DIR__ . '/model_cache.json', json_encode([
        'classCounts' => $nb->classCounts ?? [],
        'featureCounts' => $nb->featureCounts ?? [],
        'totalSamples' => $nb->totalSamples ?? 0,
        'features' => $features,
        'trained_at' => date('c'),
        'sample_count' => count($data),
    ]));

    echo json_encode(['success' => true, 'samples' => count($data), 'features' => $features]);
    exit;
}

if ($action === 'predict') {
    $body = json_decode(file_get_contents('php://input'), true);
    $input = $body['symptoms'] ?? [];

    if (empty($input)) {
        echo json_encode(['error' => 'No symptoms provided']);
        exit;
    }

    // Try cache first, else train on-the-fly
    $cacheFile = __DIR__ . '/model_cache.json';
    $csvPath = __DIR__ . '/disease_data.csv';

    if (!file_exists($cacheFile)) {
        if (!file_exists($csvPath)) {
            // Use demo predictions when no CSV is present
            $demoResults = predictDemo($input);
            echo json_encode(['predictions' => $demoResults, 'mode' => 'demo']);
            exit;
        }
        // Train and cache
        [$headers, $data] = loadCSV($csvPath);
        $features = array_values(array_filter($headers, fn($h) => $h !== 'label' && $h !== 'duration'));
        $nb = new NaiveBayesClassifier();
        $nb->train($data, $features);
        $model = [
            'classCounts' => (array)(new ReflectionProperty($nb, 'classCounts'))->getValue($nb),
            'featureCounts' => (array)(new ReflectionProperty($nb, 'featureCounts'))->getValue($nb),
            'totalSamples' => (new ReflectionProperty($nb, 'totalSamples'))->getValue($nb),
            'features' => $features,
        ];
        file_put_contents($cacheFile, json_encode($model));
    }

    $model = json_decode(file_get_contents($cacheFile), true);

    // Reconstruct classifier from cache
    $nb = new NaiveBayesClassifier();
    $nbRef = new ReflectionClass($nb);

    $cc = $nbRef->getProperty('classCounts'); $cc->setAccessible(true); $cc->setValue($nb, $model['classCounts']);
    $fc = $nbRef->getProperty('featureCounts'); $fc->setAccessible(true); $fc->setValue($nb, $model['featureCounts']);
    $ts = $nbRef->getProperty('totalSamples'); $ts->setAccessible(true); $ts->setValue($nb, $model['totalSamples']);
    $ff = $nbRef->getProperty('features'); $ff->setAccessible(true); $ff->setValue($nb, $model['features']);

    $predictions = $nb->predict($input);

    // Top 3
    $top3 = array_slice($predictions, 0, 3, true);

    echo json_encode([
        'predictions' => $top3,
        'mode' => 'trained',
        'model_info' => [
            'sample_count' => $model['totalSamples'],
            'trained_at' => $model['trained_at'] ?? null,
        ]
    ]);
    exit;
}

// ─── Demo mode (no CSV) ───────────────────────────────────────────────────────

function predictDemo(array $input): array {
    $symptomMap = [
        'Dengue'    => ['fever', 'mosquito_bite', 'rash', 'headache', 'appetite_loss'],
        'Flu'       => ['fever', 'cough', 'runny_nose', 'headache', 'sore_throat'],
        'Pneumonia' => ['fever', 'cough', 'breathing_difficulty', 'chest_pain'],
        'Malaria'   => ['fever', 'vomiting', 'headache', 'mosquito_bite'],
        'Typhoid'   => ['fever', 'appetite_loss', 'vomiting', 'headache'],
    ];

    $scores = [];
    foreach ($symptomMap as $disease => $symptoms) {
        $match = 0;
        foreach ($symptoms as $s) {
            if (isset($input[$s]) && ($input[$s] === 'Ya' || $input[$s] === 'high' || $input[$s] === 'severe')) {
                $match++;
            }
        }
        $scores[$disease] = round(($match / count($symptoms)) * 100, 1);
    }

    arsort($scores);
    return array_slice($scores, 0, 3, true);
}

echo json_encode(['error' => 'Unknown action']);
