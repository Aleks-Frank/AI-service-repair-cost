import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Animated, Easing, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Header, Screen } from '../components/ui';
import type { ScreenProps } from '../navigation/types';

type StepState = 'pending' | 'running' | 'done';

const DARK = {
  bg: '#111827',
  card: '#1F2937',
  border: '#374151',
  text: '#F9FAFB',
  muted: '#9CA3AF',
  green: '#10B981',
  blue: '#3B82F6',
};

export function Processing({ navigation }: ScreenProps<'Processing'>) {
  const [cv, setCv] = useState<StepState>('running');
  const [nlp, setNlp] = useState<StepState>('pending');
  const [rag, setRag] = useState<StepState>('pending');
  const [ragDone, setRagDone] = useState(false);
  const progress = useMemo(() => new Animated.Value(0), []);

  useEffect(() => {
    const timers = [
      setTimeout(() => setCv('done'), 1100),
      setTimeout(() => setNlp('running'), 1300),
      setTimeout(() => setNlp('done'), 2300),
      setTimeout(() => setRag('running'), 2500),
      setTimeout(() => {
        setRag('done');
        setRagDone(true);
      }, 3700),
      setTimeout(() => navigation.replace('Estimate'), 4300),
    ];
    Animated.timing(progress, { toValue: 1, duration: 4200, easing: Easing.linear, useNativeDriver: false }).start();
    return () => timers.forEach(clearTimeout);
  }, [navigation, progress]);

  return (
    <Screen>
      <View style={styles.dark}>
        <Header title="Обработка" onBack={() => navigation.goBack()} dark />
        <Text style={styles.subtitle}>AI-агент строит черновик сметы</Text>

        <Step
          icon="eye-outline"
          title="Анализ фото"
          state={cv}
          results={cv !== 'pending' ? [{ label: 'Тип дефекта', value: 'Скол плитки' }, { label: 'Размер', value: '0.5 м²' }] : []}
        />
        <Step
          icon="chatbubble-ellipses-outline"
          title="Извлечение работ из текста"
          state={nlp}
          results={nlp !== 'pending' ? [{ label: 'Работы', value: 'Замена плитки' }, { label: '', value: 'Затирка швов' }] : []}
        />
        <Step icon="server-outline" title="Поиск расценок в ТЕР / ФЕР" state={rag} results={ragDone ? [{ label: 'Найдено', value: '3 расценки • 5 материалов' }] : []} />

        <View style={styles.infoCard}>
          <Ionicons name="sparkles" size={16} color={DARK.blue} />
          <Text style={styles.infoText}>ИИ подбирает аналоги из векторной базы знаний…</Text>
        </View>

        <View style={styles.progressWrap}>
          <View style={styles.progressTrack}>
            <Animated.View style={[styles.progressFill, { width: progress.interpolate({ inputRange: [0, 1], outputRange: ['4%', '100%'] }) }]} />
          </View>
          <Text style={styles.progressLabel}>Сметы</Text>
        </View>
      </View>
    </Screen>
  );
}

function Step({
  icon,
  title,
  state,
  results,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  state: StepState;
  results: { label: string; value: string }[];
}) {
  return (
    <View style={[styles.stepCard, state === 'running' && styles.stepRunning]}>
      <View style={styles.stepHead}>
        <View style={styles.stepIcon}>
          <Ionicons name={icon} size={20} color={state === 'done' ? DARK.green : DARK.blue} />
        </View>
        <Text style={styles.stepTitle}>{title}</Text>
        <View style={{ marginLeft: 'auto' }}>
          {state === 'done' ? (
            <Ionicons name="checkmark-circle" size={22} color={DARK.green} />
          ) : state === 'running' ? (
            <ActivityIndicator size="small" color={DARK.blue} />
          ) : (
            <Ionicons name="ellipse-outline" size={20} color={DARK.border} />
          )}
        </View>
      </View>
      {state === 'running' ? <Text style={styles.runningText}>Выполняется…</Text> : null}
      {results.length ? (
        <View style={styles.results}>
          {results.map((r, i) => (
            <View key={i} style={styles.resultRow}>
              {r.label ? <Text style={styles.resultLabel}>{r.label}:</Text> : <Text style={styles.resultLabel}>{' '.repeat(1)}</Text>}
              <Text style={styles.resultValue}>{r.value}</Text>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  dark: { flex: 1, backgroundColor: DARK.bg, paddingHorizontal: 16 },
  subtitle: { color: DARK.muted, fontSize: 13, textAlign: 'center', marginBottom: 18, marginTop: -4 },
  stepCard: {
    backgroundColor: DARK.card,
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: DARK.border,
  },
  stepRunning: { borderColor: DARK.blue },
  stepHead: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  stepIcon: {
    width: 38,
    height: 38,
    borderRadius: 10,
    backgroundColor: 'rgba(59,130,246,0.12)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepTitle: { color: DARK.text, fontSize: 15, fontWeight: '700' },
  runningText: { color: DARK.blue, fontSize: 12, marginTop: 8, marginLeft: 48 },
  results: { marginTop: 10, marginLeft: 48, gap: 4 },
  resultRow: { flexDirection: 'row', gap: 6, alignItems: 'center' },
  resultLabel: { color: DARK.muted, fontSize: 12.5 },
  resultValue: { color: DARK.green, fontSize: 12.5, fontWeight: '700' },
  infoCard: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
    backgroundColor: 'rgba(59,130,246,0.10)',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(59,130,246,0.35)',
    padding: 14,
    marginTop: 6,
  },
  infoText: { color: DARK.text, fontSize: 13, flex: 1 },
  progressWrap: { marginTop: 22 },
  progressTrack: { height: 8, borderRadius: 4, backgroundColor: DARK.border, overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: 4, backgroundColor: DARK.blue },
  progressLabel: { color: DARK.muted, fontSize: 12, marginTop: 8, textAlign: 'right' },
});
