import React, { useEffect, useMemo, useState } from 'react';
import { Alert, Animated, Easing, Image, Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { AiBadge, Button, Card, Header, Screen } from '../components/ui';
import { colors, radius } from '../theme';
import { imgSource, useStore } from '../store';
import type { ScreenProps } from '../navigation/types';

const CV_HINTS = [
  { kw: ['трещ', 'фасад'], text: 'CV: обнаружена трещина, ширина ~2 мм' },
  { kw: ['плитк', 'скол'], text: 'CV: скол плитки, площадь ~0.5 м²' },
  { kw: ['теч', 'вод', 'паркинг'], text: 'CV: следы протечки, вероятна неисправность трубы' },
  { kw: ['доводчик', 'двер'], text: 'CV: доводчик ослаблен, провисание створки' },
];

const DEFAULT_CV = 'CV: дефект распознан, тип уточняется';

export function NewRequest({ navigation }: ScreenProps<'NewRequest'>) {
  const { draft, setDraft, houses } = useStore();
  const [houseModal, setHouseModal] = useState(false);
  const [listening, setListening] = useState(false);
  const pulse = useMemo(() => new Animated.Value(0), []);

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 600, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0, duration: 600, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    if (listening) loop.start();
    else {
      loop.stop();
      pulse.setValue(0);
    }
    return () => loop.stop();
  }, [listening, pulse]);

  const pickPhoto = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.7 });
    if (!res.canceled && res.assets[0]) setDraft({ photoUri: res.assets[0].uri });
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Нет доступа', 'Разрешите доступ к камере в настройках.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({ quality: 0.7 });
    if (!res.canceled && res.assets[0]) setDraft({ photoUri: res.assets[0].uri });
  };

  const cvText = CV_HINTS.find((h) => h.kw.some((k) => draft.description.toLowerCase().includes(k)))?.text ?? DEFAULT_CV;
  const canGenerate = Boolean(draft.photoUri) && draft.description.trim().length >= 8;

  return (
    <Screen>
      <Header title="Новый дефект" onBack={() => navigation.goBack()} />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.sectionLabel}>Фото дефекта</Text>
        {draft.photoUri ? (
          <View>
            <Image source={imgSource(draft.photoUri) as never} style={styles.photoPreview} resizeMode="cover" />
            <View style={styles.cvFloat}>
              <AiBadge text={cvText} />
            </View>
            <Pressable onPress={() => setDraft({ photoUri: null })} style={styles.removePhoto}>
              <Ionicons name="trash-outline" size={16} color={colors.danger} />
              <Text style={{ color: colors.danger, fontSize: 12, fontWeight: '600', marginLeft: 4 }}>Удалить фото</Text>
            </Pressable>
          </View>
        ) : (
          <Pressable style={styles.dropZone} onPress={pickPhoto}>
            <Ionicons name="camera-outline" size={36} color={colors.primary} />
            <Text style={styles.dropTitle}>Сделать фото или выбрать из галереи</Text>
            <Text style={styles.dropHint}>PNG / JPG, до 10 МБ</Text>
          </Pressable>
        )}
        {!draft.photoUri ? (
          <View style={styles.photoBtnRow}>
            <Button label="Сделать фото" icon="camera" variant="outline" onPress={takePhoto} style={{ flex: 1, marginTop: 10 }} />
            <Button
              label="Демо"
              icon="flask-outline"
              variant="ghost"
              onPress={() => setDraft({ photoUri: require('../../assets/tile-chip.png'), description: 'Скол керамической плитки на полу в коридоре, около 3 плиток затёрты' })}
              style={{ flex: 1, marginTop: 10 }}
            />
          </View>
        ) : null}

        <Text style={styles.sectionLabel}>Описание</Text>
        <View style={styles.inputWrap}>
          <TextInput
            style={styles.input}
            multiline
            placeholder='Опишите проблему (например: «Течь в паркинге, мокнет стена 3 метра»)'
            placeholderTextColor={colors.textMuted}
            value={draft.description}
            onChangeText={(t) => setDraft({ description: t })}
          />
          <Pressable
            onPress={() => setListening((v) => !v)}
            style={[styles.micBtn, listening && { backgroundColor: colors.warning }]}
            hitSlop={8}
          >
            <Animated.View
              style={{
                opacity: pulse.interpolate({ inputRange: [0, 1], outputRange: [0.6, 1] }),
                transform: [{ scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.12] }) }],
              }}
            >
              <Ionicons name={listening ? 'mic' : 'mic-outline'} size={20} color={listening ? '#FFF' : colors.primary} />
            </Animated.View>
          </Pressable>
        </View>
        {listening ? <Text style={styles.listeningText}>Слушаю… говорите описание дефекта</Text> : null}

        <Text style={styles.sectionLabel}>Характеристики дома</Text>
        <Pressable onPress={() => setHouseModal(true)}>
          <Card style={styles.houseCard}>
            <View style={styles.houseIcon}>
              <Ionicons name="business-outline" size={22} color={colors.primary} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.houseTitle}>{draft.house}</Text>
              <View style={styles.passportLine}>
                <Ionicons name="checkmark-circle" size={14} color={colors.success} />
                <Text style={styles.passportText}>Паспорт дома загружен</Text>
              </View>
            </View>
            <Ionicons name="chevron-forward" size={20} color={colors.textMuted} />
          </Card>
        </Pressable>

        <Button
          label="Сгенерировать смету"
          icon="flash"
          disabled={!canGenerate}
          onPress={() => navigation.navigate('Processing')}
          style={[styles.cta, canGenerate && styles.ctaGlow]}
        />
        {!canGenerate ? (
          <Text style={styles.hintText}>Добавьте фото и опишите дефект — этого достаточно для черновика.</Text>
        ) : null}
        <View style={{ height: 40 }} />
      </ScrollView>

      <Modal visible={houseModal} transparent animationType="fade" onRequestClose={() => setHouseModal(false)}>
        <Pressable style={styles.backdrop} onPress={() => setHouseModal(false)}>
          <Card style={styles.sheet}>
            <Text style={styles.sheetTitle}>Выберите объект</Text>
            {houses.map((h) => (
              <Pressable
                key={h}
                style={[styles.sheetRow, h === draft.house && styles.sheetRowActive]}
                onPress={() => {
                  setDraft({ house: h });
                  setHouseModal(false);
                }}
              >
                <Text style={[styles.sheetRowText, h === draft.house && { color: '#FFF', fontWeight: '700' }]}>{h}</Text>
                {h === draft.house ? <Ionicons name="checkmark" size={18} color="#FFF" /> : null}
              </Pressable>
            ))}
          </Card>
        </Pressable>
      </Modal>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: { paddingHorizontal: 16, paddingTop: 4, flexGrow: 1 },
  sectionLabel: { fontSize: 13, fontWeight: '700', color: colors.textMuted, marginTop: 18, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.4 },
  dropZone: {
    borderWidth: 1.6,
    borderStyle: 'dashed',
    borderColor: colors.primary,
    borderRadius: radius.card,
    backgroundColor: '#EFF6FF',
    paddingVertical: 34,
    alignItems: 'center',
  },
  dropTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginTop: 10 },
  dropHint: { fontSize: 12, color: colors.textMuted, marginTop: 4 },
  photoPreview: { width: '100%', height: 220, borderRadius: radius.card, backgroundColor: colors.border },
  cvFloat: { marginTop: -46, marginLeft: 12, marginBottom: 8 },
  removePhoto: { flexDirection: 'row', alignSelf: 'flex-end', padding: 8 },
  photoBtnRow: { flexDirection: 'row', gap: 10 },
  inputWrap: { position: 'relative' },
  input: {
    backgroundColor: colors.surface,
    borderRadius: radius.input,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 14,
    paddingRight: 52,
    minHeight: 92,
    textAlignVertical: 'top',
    fontSize: 14.5,
    color: colors.text,
  },
  micBtn: {
    position: 'absolute',
    right: 10,
    bottom: 10,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#EFF6FF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  listeningText: { color: '#B45309', fontSize: 12.5, fontWeight: '600', marginTop: 6 },
  houseCard: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  houseIcon: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#EFF6FF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  houseTitle: { fontSize: 14.5, fontWeight: '700', color: colors.text },
  passportLine: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 4 },
  passportText: { fontSize: 12, color: colors.successDark, fontWeight: '600' },
  cta: { marginTop: 26 },
  ctaGlow: {
    shadowColor: colors.primary,
    shadowOpacity: 0.35,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  hintText: { textAlign: 'center', color: colors.textMuted, fontSize: 12, marginTop: 10 },
  backdrop: { flex: 1, backgroundColor: 'rgba(17,24,39,0.5)', justifyContent: 'flex-end' },
  sheet: { marginBottom: 18, marginHorizontal: 12 },
  sheetTitle: { fontSize: 16, fontWeight: '800', marginBottom: 12, color: colors.text },
  sheetRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 13,
    paddingHorizontal: 14,
    borderRadius: 12,
    backgroundColor: colors.bg,
    marginBottom: 8,
  },
  sheetRowActive: { backgroundColor: colors.primary },
  sheetRowText: { fontSize: 14.5, color: colors.text },
});
